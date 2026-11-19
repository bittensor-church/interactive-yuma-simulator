from django.test import RequestFactory

from project.core.views import simulate_single_case_view
from project.yuma_simulation._internal.cases import cases
from project.yuma_simulation._internal.yumas import YumaSimulationNames

MARKUP = "<b>probe</b>"


def _get(params: dict[str, str]):
    return simulate_single_case_view(RequestFactory().get("/simulate_single_case/", params))


def _assert_plain_text_and_not_cached(response):
    # Errors echo request parameters: they must never be rendered as HTML ...
    assert response["Content-Type"].startswith("text/plain")
    # ... and a failure must not be cached like a simulation result.
    cache_control = response["Cache-Control"]
    assert "no-store" in cache_control
    assert "public" not in cache_control


def test__single_case__unknown_case_is_a_plain_text_uncached_404():
    response = _get({"case_name": MARKUP})

    assert response.status_code == 404
    assert MARKUP in response.content.decode()
    _assert_plain_text_and_not_cached(response)


def test__single_case__invalid_number_is_a_plain_text_uncached_400():
    response = _get({"case_name": cases[0].name, "kappa": MARKUP})

    assert response.status_code == 400
    _assert_plain_text_and_not_cached(response)


def test__single_case__missing_case_name_is_a_plain_text_uncached_400():
    response = _get({})

    assert response.status_code == 400
    _assert_plain_text_and_not_cached(response)


def test__single_case__successful_simulation_is_html_and_publicly_cached():
    response = _get({"case_name": cases[0].name, "chosen_yuma": YumaSimulationNames().YUMA2})

    assert response.status_code == 200
    assert response["Content-Type"].startswith("text/html")
    assert "<img" in response.content.decode()
    cache_control = response["Cache-Control"]
    assert "public" in cache_control
    assert "max-age=604800" in cache_control
    assert "no-store" not in cache_control


# --- parameter validation: single case ----------------------------------------------------


def test__single_case__rejects_values_outside_the_form_limits():
    for params in (
        {"kappa": "nan"},
        {"kappa": "inf"},
        {"kappa": "70000"},
        {"bond_penalty": "-1"},
        {"bond_moving_avg": "1000001"},
        {"alpha_tao_ratio": "-0.1"},
        {"liquid_alpha_consensus_mode": "SOMETHING"},
    ):
        response = _get({"case_name": cases[0].name, **params})

        assert response.status_code == 400, params
        _assert_plain_text_and_not_cached(response)


def test__single_case__unknown_yuma_version_is_rejected_before_simulating():
    response = _get({"case_name": cases[0].name, "chosen_yuma": "Yuma 99"})

    assert response.status_code == 400
    _assert_plain_text_and_not_cached(response)
