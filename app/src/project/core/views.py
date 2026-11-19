import json
import logging
import math
import re
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from functools import wraps
from django.conf import settings


import pandas as pd
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.utils.cache import add_never_cache_headers, patch_cache_control
from requests.exceptions import HTTPError, Timeout
from project.yuma_simulation._internal.cases import get_synthetic_cases, MetagraphCase
from project.yuma_simulation._internal.yumas import SimulationHyperparameters, YumaParams, YumaSimulationNames
from project.yuma_simulation.v1 import api as yuma_api
from project.yuma_simulation.v1.api import generate_chart_table, generate_metagraph_based_chart_table

from .forms import MAX_METAGRAPH_RANGE, SelectionForm, SimulationHyperparametersForm, YumaParamsForm
from .utils import ONE_MILLION, UINT16_MAX, fetch_metagraph_data, normalize

logger = logging.getLogger(__name__)

RESULT_CACHE_SECONDS = 7 * 24 * 60 * 60


def _error_response(message: str, status: int) -> HttpResponse:
    """
    Error responses echo request parameters, so they are sent as plain text:
    as text/html the browser would render markup taken from the URL.
    """
    return HttpResponse(message, status=status, content_type="text/plain; charset=utf-8")


def cache_successful_responses(view):
    """
    Let browsers and proxies cache simulation results, but only successful ones.
    A failure (upstream timeout, missing data, bad parameter) must not stick for a week.
    """

    @wraps(view)
    def wrapper(request, *args, **kwargs):
        response = view(request, *args, **kwargs)
        if response.status_code == 200:
            patch_cache_control(
                response,
                public=True,
                max_age=RESULT_CACHE_SECONDS,
                s_maxage=RESULT_CACHE_SECONDS,
            )
        else:
            add_never_cache_headers(response)
        return response

    return wrapper


def simulation_view(request):
    selection_form = SelectionForm(request.GET or None)
    hyper_form = SimulationHyperparametersForm(request.GET or None)
    yuma_form = YumaParamsForm(request.GET or None, selected_yuma=request.GET.get('selected_yumas'),)

    context = {
        "selection_form": selection_form,
        "hyper_form": hyper_form,
        "yuma_form": yuma_form,
        "valid_forms": False,
        "cases_json": "[]",
        "yumas_json": "{}",
    }

    if request.GET.get("use_metagraph"):
        selection_form.fields["selected_cases"].required = False

    if selection_form.is_valid() and hyper_form.is_valid() and yuma_form.is_valid():
        cleaned = selection_form.cleaned_data

        if selection_form.is_valid():
            context["valid_forms"] = True

            selected_case_names = cleaned.get("selected_cases", [])
            context["cases_json"] = json.dumps(selected_case_names)

            yumas_dict = asdict(YumaSimulationNames())
            yuma_key = cleaned["selected_yumas"]
            yuma_data = yuma_form.cleaned_data

            if yuma_key == "YUMA3" and yuma_data["liquid_alpha"]:
                yuma_key += "_LIQUID"

            chosen_yuma = yumas_dict[yuma_key]
            hyper_data = hyper_form.cleaned_data

            sim_params = {
                "kappa": hyper_data["kappa"],
                "bond_penalty": hyper_data["bond_penalty"],
                "reset_bonds": hyper_data["reset_bonds"],
                "liquid_alpha_consensus_mode": hyper_data["liquid_alpha_consensus_mode"],
                "alpha_tao_ratio": hyper_data["alpha_tao_ratio"],
            }
            yuma_params = {
                "bond_moving_avg": yuma_data["bond_moving_avg"],
                "liquid_alpha": yuma_data["liquid_alpha"],
                "alpha_high": yuma_data["alpha_high"],
                "alpha_low": yuma_data["alpha_low"],
                "decay_rate": yuma_data["decay_rate"],
                "capacity_alpha": yuma_data["capacity_alpha"],
                "alpha_sigmoid_steepness": yuma_data["alpha_sigmoid_steepness"],
            }

            effective_weights = selection_form.cleaned_data.get("liquid_alpha_effective_weights", False)
            if yuma_key.startswith("YUMA3"):
                yuma_params["liquid_alpha_effective_weights"] = effective_weights

            context["yumas_json"] = json.dumps(
                {
                    "selected_yuma_key": yuma_key,
                    "chosen_yuma": chosen_yuma,
                    "shifted_validator_hotkey": cleaned["shifted_validator_hotkey"],
                    "sim_params": sim_params,
                    "yuma_params": yuma_params,
                }
            )
            context["cache_key"] = settings.CACHE_KEY
            context["jsCharts"] = request.GET.get('jsCharts', 'off')

    return render(request, "simulator.html", context)


class InvalidParameter(ValueError):
    """A request parameter is missing, malformed or out of range."""


YUMA_NAMES_BY_KEY: dict[str, str] = asdict(YumaSimulationNames())
CONSENSUS_MODES = [value for value, _label in SimulationHyperparametersForm.base_fields["liquid_alpha_consensus_mode"].choices]


def _float_param(request, name: str, default: float, *, min_value: float | None = None, max_value: float | None = None) -> float:
    raw = request.GET.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        raise InvalidParameter(f"'{name}' must be a number.")
    if not math.isfinite(value):
        raise InvalidParameter(f"'{name}' must be a finite number.")
    if min_value is not None and value < min_value:
        raise InvalidParameter(f"'{name}' must be at least {min_value:g}.")
    if max_value is not None and value > max_value:
        raise InvalidParameter(f"'{name}' must be at most {max_value:g}.")
    return value


def _bool_param(request, name: str) -> bool:
    return request.GET.get(name, "False") == "true"


def _choice_param(request, name: str, default: str, choices) -> str:
    value = request.GET.get(name, default)
    if value not in choices:
        raise InvalidParameter(f"'{name}' is not one of the supported values.")
    return value


def _datetime_param(request, name: str) -> datetime:
    """Parse an ISO datetime; the result is naive and in UTC, like the values the form produces."""
    raw = request.GET.get(name)
    if not raw:
        raise InvalidParameter(f"'{name}' is required.")
    try:
        value = datetime.fromisoformat(raw)
    except ValueError:
        raise InvalidParameter(f"'{name}' must be an ISO date, e.g. 2025-01-31T12:00:00.")
    if value.tzinfo is not None:
        value = value.astimezone(UTC).replace(tzinfo=None)
    return value


def _parse_hyperparameters(request) -> dict:
    """
    Limits mirror SimulationHyperparametersForm and YumaParamsForm: the page validates the
    same values with those forms, but these endpoints can also be called directly.
    """
    return {
        "kappa": _float_param(request, "kappa", 32767, min_value=0, max_value=UINT16_MAX),
        "bond_penalty": _float_param(request, "bond_penalty", 65535, min_value=0, max_value=UINT16_MAX),
        "liquid_alpha_consensus_mode": _choice_param(request, "liquid_alpha_consensus_mode", "CURRENT", CONSENSUS_MODES),
        "alpha_tao_ratio": _float_param(request, "alpha_tao_ratio", 0.1, min_value=0),
        "bond_moving_avg": _float_param(request, "bond_moving_avg", 900_000, min_value=0, max_value=ONE_MILLION),
        "liquid_alpha": _bool_param(request, "liquid_alpha"),
        "liquid_alpha_effective_weights": _bool_param(request, "liquid_alpha_effective_weights"),
        "alpha_high": _float_param(request, "alpha_high", 0.3),
        "alpha_low": _float_param(request, "alpha_low", 0.1),
        "decay_rate": _float_param(request, "decay_rate", 0.1),
        "capacity_alpha": _float_param(request, "capacity_alpha", 0.1),
        "alpha_sigmoid_steepness": _float_param(request, "alpha_sigmoid_steepness", 10.0),
    }


def _yuma_kwargs(params: dict) -> dict:
    return {
        "bond_moving_avg": normalize(params["bond_moving_avg"], ONE_MILLION),
        "liquid_alpha": params["liquid_alpha"],
        "alpha_high": params["alpha_high"],
        "alpha_low": params["alpha_low"],
        "decay_rate": params["decay_rate"],
        "capacity_alpha": params["capacity_alpha"],
        "alpha_sigmoid_steepness": params["alpha_sigmoid_steepness"],
    }


@cache_successful_responses
def simulate_single_case_view(request):
    """
    Returns HTML snippet for a single case simulation.
    Expects query parameters for case_name, sim_params, yuma_params, etc.
    """
    case_name = request.GET.get("case_name")
    if not case_name:
        return _error_response("Missing 'case_name' parameter.", status=400)

    try:
        params = _parse_hyperparameters(request)
        # this endpoint takes the display name of the Yuma version, not its key
        chosen_yuma = _choice_param(
            request, "chosen_yuma", YUMA_NAMES_BY_KEY["YUMA2"], list(YUMA_NAMES_BY_KEY.values())
        )
    except InvalidParameter as e:
        return _error_response(f"Invalid parameter: {e}", status=400)

    reset_bonds = _bool_param(request, "reset_bonds")
    js_charts = request.GET.get('jsCharts', '') == 'on'

    sim_params = SimulationHyperparameters(
        kappa=normalize(params["kappa"], UINT16_MAX),
        bond_penalty=normalize(params["bond_penalty"], UINT16_MAX),
        liquid_alpha_consensus_mode=params["liquid_alpha_consensus_mode"],
    )

    yuma_kwargs = _yuma_kwargs(params)
    if chosen_yuma.startswith("YUMA3"):
        yuma_kwargs["liquid_alpha_effective_weights"] = params["liquid_alpha_effective_weights"]

    yuma_params = YumaParams(**yuma_kwargs)

    package_cases = get_synthetic_cases(use_full_matrices=True, reset_bonds=reset_bonds)
    chosen_case = next((c for c in package_cases if c.name == case_name), None)
    if not chosen_case:
        return _error_response(f"Case '{case_name}' not found.", status=404)

    try:
        selected_yumas = [(chosen_yuma, yuma_params)]
        partial_html = generate_chart_table(
            cases=[chosen_case],
            yuma_versions=selected_yumas,
            yuma_hyperparameters=sim_params,
            engine='plotly' if js_charts else 'matplotlib',
        )
    except Exception as e:
        logger.exception("Error generating chart for case %r", case_name)
        return _error_response(f"Error generating chart: {e}", status=500)

    return HttpResponse(partial_html.data if partial_html else "No data", status=200)


@cache_successful_responses
def metagraph_simulation_view(request):
    try:
        params = _parse_hyperparameters(request)
        chosen_yuma = _choice_param(request, "selected_yumas", "YUMA2", list(YUMA_NAMES_BY_KEY))

        # same rules as SelectionForm.clean()
        start_date = _datetime_param(request, "start_date")
        end_date = _datetime_param(request, "end_date")
        now = datetime.now(UTC).replace(tzinfo=None)
        if end_date < start_date:
            raise InvalidParameter("'end_date' cannot be before 'start_date'.")
        if end_date > now:
            raise InvalidParameter("'end_date' cannot be in the future.")
        if end_date > start_date + MAX_METAGRAPH_RANGE:
            raise InvalidParameter(
                f"'end_date' cannot be more than {MAX_METAGRAPH_RANGE.days} days after 'start_date'."
            )

        try:
            netuid = int(request.GET.get("netuid", 0))
        except ValueError:
            raise InvalidParameter("'netuid' must be an integer.")
        if netuid < 0:
            raise InvalidParameter("'netuid' must not be negative.")
    except InvalidParameter as e:
        return _error_response(f"Invalid parameter: {e}", status=400)

    js_charts = request.GET.get('jsCharts', '') == 'on'
    shifted_validator_hotkey = request.GET.get('shifted_validator_hotkey', '')
    requested_miners = [m.strip() for m in request.GET.getlist("miners_hotkeys") if m.strip()]

    sim_params = SimulationHyperparameters(
        kappa=normalize(params["kappa"], UINT16_MAX),
        bond_penalty=normalize(params["bond_penalty"], UINT16_MAX),
        liquid_alpha_consensus_mode=params["liquid_alpha_consensus_mode"],
        alpha_tao_ratio=params["alpha_tao_ratio"],
    )

    mg_yuma_kwargs = _yuma_kwargs(params)
    if chosen_yuma.startswith("YUMA3"):
        mg_yuma_kwargs["liquid_alpha_effective_weights"] = params["liquid_alpha_effective_weights"]

    yuma_params = YumaParams(**mg_yuma_kwargs)

    epochs_padding = int(settings.EPOCHS_PADDING)
    start_date = start_date - timedelta(seconds=360 * 12 * epochs_padding)
    try:
        mg_data = fetch_metagraph_data(
            start_date=start_date,
            end_date=end_date,
            netuid=netuid,
        )
    except Timeout:
        return _error_response(
            "The service timed out. Try requesting fewer epochs.",
            status=500,
        )
    except HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return _error_response(f"No metagraph data for timespan {start_date}–{end_date}", status=404)
        if e.response is not None and e.response.status_code >= 500:
            return _error_response(
                "Internal Server Error. "
                "Make sure you’re querying historical metagraph data no older than 35 days ago.",
                status=500,
            )
        return _error_response(f"Error fetching metagraph data: {e}", status=400)
    except Exception as e:
        logger.exception("Error fetching metagraph data")
        return _error_response(f"Error fetching metagraph data: {e}", status=400)

    case_config = {}
    if shifted_validator_hotkey:
        case_config.update({
            'introduce_shift': True,
            'shift_validator_hotkey': shifted_validator_hotkey,
        })
    try:
        case, invalid_miners = MetagraphCase.from_mg_dumper_data(
            mg_data=mg_data,
            requested_miners=requested_miners,
            **case_config,
            )
    except ValueError as e:
        return _error_response(str(e), status=400)
    except (KeyError, IndexError, TypeError):
        # the metagraph service answered, but not with the structure the simulation expects
        logger.exception("Unexpected metagraph data for netuid %s", netuid)
        return _error_response("The metagraph data for this timespan could not be processed.", status=500)

    selection_form = SelectionForm(request.GET or None)
    for hotkey in invalid_miners:
        selection_form.add_error(
            "miners_hotkeys",
            f"Invalid miner hotkey: {hotkey}"
        )

    selected_chart_yumas = [(YUMA_NAMES_BY_KEY[chosen_yuma], yuma_params)]
    all_yumas = list(YUMA_NAMES_BY_KEY.values())
    summary_versions = [
        (display_name, yuma_params)
        for display_name in all_yumas
        if display_name not in ("Yuma 1", "Yuma 3 (Rhef+relative bonds)")
    ]

    try:
        partial_html = generate_metagraph_based_chart_table(
            chart_versions=selected_chart_yumas,
            summary_versions=summary_versions,
            normal_case=case,
            yuma_hyperparameters=sim_params,
            epochs_padding=epochs_padding,
            engine='plotly' if js_charts else 'matplotlib',
        )
    except Exception as e:
        logger.exception("Error generating metagraph charts for netuid %s", netuid)
        return _error_response(f"Error generating chart: {e}", status=500)

    return JsonResponse(
        {
            "html": partial_html.data or "No data",
            "errors": selection_form.errors,
        },
        status=200
    )

def bootstrap_generate_ipynb_table(
    table_data: dict[str, list[str]],
    summary_table: pd.DataFrame | None,
    case_row_ranges: list[tuple[int, int, int]],
) -> str:
    if summary_table is None:
        summary_table = pd.DataFrame(table_data)

    cols = list(summary_table.columns)
    rows_html: list[str] = []

    def parse_img_src(html_str: str) -> str:
        m = re.search(r'src="([^"]+)"', html_str)
        return m.group(1) if m else ""

    for i in range(len(summary_table)):
        # find a matching (start,end,c_idx) where c_idx is in-bounds
        match = next(
            ((start, end, c_idx) for start, end, c_idx in case_row_ranges
             if start <= i <= end and 0 <= c_idx < len(cols)),
            None
        )
        if match:
            _, _, c_idx = match
        else:
            # fallback to the first column
            c_idx = 0

        case_name = cols[c_idx]
        raw_html = summary_table.iat[i, c_idx]
        img_src = parse_img_src(raw_html)

        if img_src:
            rows_html.append(f'''
              <div class="mb-4 text-center">
                <img src="{img_src}"
                     class="img-fluid w-100"
                     alt="Chart for {case_name}">
              </div>
            ''')
        else:
            rows_html.append(f'''
              <div class="mb-4 text-center">
                {raw_html}
              </div>
            ''')

    return f'''
    <div class="container-fluid px-3">
      {''.join(rows_html)}
    </div>
    '''


yuma_api._generate_ipynb_table = bootstrap_generate_ipynb_table
