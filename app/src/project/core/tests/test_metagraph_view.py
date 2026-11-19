from datetime import UTC, datetime, timedelta
from unittest import mock

import pytest
import requests
from django.test import RequestFactory

from project.core import utils, views


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None, microsecond=0)


def _valid_params() -> dict[str, str]:
    end = _now() - timedelta(hours=1)
    return {
        "start_date": (end - timedelta(days=1)).isoformat(),
        "end_date": end.isoformat(),
        "netuid": "1",
        "selected_yumas": "YUMA2",
    }


def _get(params: dict[str, str]):
    return views.metagraph_simulation_view(RequestFactory().get("/simulate_metagraph/", params))


def _assert_plain_text_and_not_cached(response):
    assert response["Content-Type"].startswith("text/plain")
    assert "no-store" in response["Cache-Control"]


@pytest.fixture
def fetch(monkeypatch):
    """Replace the call to the metagraph service; fails the test if a rejected request reaches it."""
    fake = mock.Mock(side_effect=AssertionError("the metagraph service must not be called"))
    monkeypatch.setattr(views, "fetch_metagraph_data", fake)
    return fake


@pytest.mark.parametrize(
    "override",
    [
        {"start_date": ""},
        {"end_date": ""},
        {"start_date": "not-a-date"},
        {"netuid": "abc"},
        {"netuid": "-1"},
        {"selected_yumas": "YUMA99"},
        {"kappa": "nan"},
        {"liquid_alpha_consensus_mode": "SOMETHING"},
    ],
)
def test__metagraph__invalid_parameters_are_rejected_without_calling_the_service(fetch, override):
    response = _get({**_valid_params(), **override})

    assert response.status_code == 400
    _assert_plain_text_and_not_cached(response)
    fetch.assert_not_called()


def test__metagraph__timespan_longer_than_the_form_allows_is_rejected(fetch):
    end = _now() - timedelta(hours=1)
    start = end - views.MAX_METAGRAPH_RANGE - timedelta(minutes=1)

    response = _get({**_valid_params(), "start_date": start.isoformat(), "end_date": end.isoformat()})

    assert response.status_code == 400
    fetch.assert_not_called()


def test__metagraph__end_in_the_future_or_before_start_is_rejected(fetch):
    future = _get({**_valid_params(), "end_date": (_now() + timedelta(hours=1)).isoformat()})
    reversed_range = _get({**_valid_params(), "start_date": _now().isoformat()})

    assert future.status_code == 400
    assert reversed_range.status_code == 400
    fetch.assert_not_called()


def test__metagraph__service_timeout_is_a_readable_uncached_error(fetch):
    fetch.side_effect = requests.exceptions.Timeout()

    response = _get(_valid_params())

    assert response.status_code == 500
    assert "timed out" in response.content.decode()
    _assert_plain_text_and_not_cached(response)


def test__metagraph__data_without_validators_is_a_400_not_a_crash(fetch):
    fetch.side_effect = None
    fetch.return_value = {
        "netuid": 1,
        "blocks": [100],
        # one registered neuron, not a validator
        "hotkeys": {"100": [["hk-miner", False, True]]},
        "weights": {"100": {}},
        "stakes": {"100": {"0": 5000.0}},
        "labels": {},
    }

    response = _get(_valid_params())

    assert response.status_code == 400
    assert "validators" in response.content.decode()
    _assert_plain_text_and_not_cached(response)


def test__metagraph__malformed_service_data_is_a_500_not_a_crash(fetch):
    fetch.side_effect = None
    fetch.return_value = {"blocks": [100], "hotkeys": {"100": []}}  # no weights / stakes

    response = _get(_valid_params())

    assert response.status_code == 500
    _assert_plain_text_and_not_cached(response)


# --- metagraph service client ---------------------------------------------------------------


def _service_response(status: int = 200, payload=None):
    response = mock.Mock(spec=requests.Response)
    response.status_code = status
    response.ok = status < 400
    response.reason = "reason"
    response.text = ""
    response.json.return_value = payload if payload is not None else {}
    return response


@pytest.fixture
def service(monkeypatch, settings):
    """A configured metagraph service whose HTTP calls are captured."""
    settings.MGRAPH_BASE_URL = "https://mg.example"
    settings.MGRAPH_API_TOKEN = "service-token"  # noqa: S105 - a fixture value, not a secret
    settings.MGRAPH_TIMEOUT = 42
    get = mock.Mock(return_value=_service_response(payload={"blocks": []}))
    monkeypatch.setattr(utils.requests, "get", get)
    return get


def test__fetch__calls_the_data_api_with_the_bearer_token_and_timeout(service):
    start = datetime(2026, 1, 1, 12, 0)
    end = datetime(2026, 1, 2, 12, 0)

    data = utils.fetch_metagraph_data(start_date=start, end_date=end, netuid=7)

    assert data == {"blocks": []}
    service.assert_called_once_with(
        "https://mg.example/api/v1/metagraph-data/",
        params={"start_date": "2026-01-01T12:00:00", "end_date": "2026-01-02T12:00:00", "netuid": 7},
        headers={"Authorization": "Bearer service-token"},
        timeout=42,
    )


@pytest.mark.parametrize("status", [401, 404, 500])
def test__fetch__raises_an_http_error_for_an_error_response(service, status):
    service.return_value = _service_response(status=status, payload={"error": "nope"})

    with pytest.raises(requests.HTTPError) as error:
        utils.fetch_metagraph_data(start_date=_now(), end_date=_now(), netuid=1)

    assert error.value.response.status_code == status


@pytest.mark.parametrize("missing", ["MGRAPH_BASE_URL", "MGRAPH_API_TOKEN"])
def test__fetch__without_configuration_does_not_call_anything(service, settings, missing):
    setattr(settings, missing, "")

    with pytest.raises(RuntimeError):
        utils.fetch_metagraph_data(start_date=_now(), end_date=_now(), netuid=1)

    service.assert_not_called()


def test__metagraph__no_data_answer_includes_the_services_explanation(fetch):
    fetch.side_effect = requests.HTTPError(
        "404 Not Found", response=_service_response(status=404, payload={"error": "No weights are stored."})
    )

    response = _get(_valid_params())

    assert response.status_code == 404
    assert "No weights are stored." in response.content.decode()
    _assert_plain_text_and_not_cached(response)


# --- summary table ---------------------------------------------------------------------------


def test__metagraph__summary_leaves_out_yuma1_and_plain_yuma3(fetch, monkeypatch):
    from project.yuma_simulation._internal.yumas import YumaSimulationNames

    names = YumaSimulationNames()
    fetch.side_effect = None
    fetch.return_value = {}
    monkeypatch.setattr(
        views.MetagraphCase, "from_mg_dumper_data", classmethod(lambda cls, **kwargs: (mock.Mock(), []))
    )
    chart_table = mock.Mock(return_value=mock.Mock(data="<div></div>"))
    monkeypatch.setattr(views, "generate_metagraph_based_chart_table", chart_table)

    response = _get({**_valid_params(), "selected_yumas": "YUMA1"})

    assert response.status_code == 200
    summary_names = [name for name, _params in chart_table.call_args.kwargs["summary_versions"]]
    assert names.YUMA1 not in summary_names
    assert names.YUMA3 not in summary_names
    assert names.YUMA2 in summary_names
    assert names.YUMA3_LIQUID in summary_names
    # the version picked for the charts is still charted, even when it is not in the summary
    assert [name for name, _params in chart_table.call_args.kwargs["chart_versions"]] == [names.YUMA1]
