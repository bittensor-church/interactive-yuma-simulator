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


def _service_response(status: int = 200, url: str = "https://mg.example/metagraph-data/", payload=None):
    response = mock.Mock(spec=requests.Response)
    response.status_code = status
    response.ok = status < 400
    response.url = url
    response.reason = "reason"
    response.text = ""
    response.json.return_value = payload if payload is not None else {}
    return response


def test__fetch__logs_in_again_when_the_cached_session_has_expired(monkeypatch, settings):
    settings.MGRAPH_BASE_URL = "https://mg.example/"
    expired = mock.Mock()
    expired.get.return_value = _service_response(url="https://mg.example/admin/login/?next=/metagraph-data/")
    fresh = mock.Mock()
    fresh.get.return_value = _service_response(payload={"blocks": []})
    get_session = mock.Mock(side_effect=[expired, fresh])
    monkeypatch.setattr(utils, "get_metagraph_session", get_session)

    data = utils.fetch_metagraph_data(start_date=_now(), end_date=_now(), netuid=1)

    assert data == {"blocks": []}
    assert get_session.call_count == 2


def test__fetch__gives_up_when_a_fresh_login_is_rejected_too(monkeypatch, settings):
    settings.MGRAPH_BASE_URL = "https://mg.example/"
    rejected = mock.Mock()
    rejected.get.return_value = _service_response(status=403)
    monkeypatch.setattr(utils, "get_metagraph_session", mock.Mock(return_value=rejected))

    with pytest.raises(RuntimeError):
        utils.fetch_metagraph_data(start_date=_now(), end_date=_now(), netuid=1)

    assert rejected.get.call_count == 2


def test__fetch__uses_the_configured_timeout(monkeypatch, settings):
    settings.MGRAPH_BASE_URL = "https://mg.example/"
    settings.MGRAPH_TIMEOUT = 42
    session = mock.Mock()
    session.get.return_value = _service_response(payload={})
    monkeypatch.setattr(utils, "get_metagraph_session", mock.Mock(return_value=session))

    utils.fetch_metagraph_data(start_date=_now(), end_date=_now(), netuid=1)

    assert session.get.call_args.kwargs["timeout"] == 42
