import logging
import re
from urllib.parse import urljoin
from datetime import datetime

import requests
import pandas as pd
from django.conf import settings
from django.core.cache import cache

UINT16_MAX = 65535.0
ONE_MILLION = 1_000_000.0

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# TODO: refactor yuma-simulation package to accept hyperparameter values natively
def normalize(value: float, max_value: float) -> float:
    """Normalize a value to the [0,1] range based on a given maximum hyperparameter value."""
    try:
        return value / max_value
    except (TypeError, ZeroDivisionError):
        raise ValueError(f"Cannot normalize value={value} with max_value={max_value}")


_CACHE_KEY = "metagraph_client_session"
_LOGIN_PATH = "admin/login/"
_LOGIN_TIMEOUT = 10


def get_metagraph_session() -> requests.Session:
    """
    Return a logged‐in Session for the external Django service.
    We cache it in Django’s cache so we only re-login once per hour.
    """
    sess = cache.get(_CACHE_KEY)
    if sess:
        return sess

    sess = requests.Session()
    login_url = urljoin(settings.MGRAPH_BASE_URL, _LOGIN_PATH)

    r1 = sess.get(login_url, timeout=_LOGIN_TIMEOUT)
    m = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', r1.text)
    if not m:
        raise RuntimeError("Could not get CSRF token")
    token = m.group(1)

    resp = sess.post(
        login_url,
        data={
            "csrfmiddlewaretoken": token,
            "username": settings.MGRAPH_USERNAME,
            "password": settings.MGRAPH_PASSWORD,
            "next": "/admin/",
        },
        headers={"Referer": login_url},
        timeout=_LOGIN_TIMEOUT,
    )
    resp.raise_for_status()
    if "admin/" not in resp.url:
        raise RuntimeError("Login failed")

    # cache for an hour (or however long your external session lives)
    cache.set(_CACHE_KEY, sess, 60 * 60)
    return sess


def _session_expired(response: requests.Response) -> bool:
    """
    The cached session can die before our own one-hour cache entry does (service restart,
    shorter session lifetime). The service then redirects to its login page or rejects the call.
    """
    return response.status_code in (401, 403) or _LOGIN_PATH in response.url


def fetch_metagraph_data(
    start_date: datetime,
    end_date: datetime,
    netuid: int,
) -> dict:
    url = urljoin(settings.MGRAPH_BASE_URL, "metagraph-data/")
    params = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "netuid": netuid,
    }

    logger.debug("→ GET %s %r", url, params)
    r = get_metagraph_session().get(url, params=params, timeout=settings.MGRAPH_TIMEOUT)

    if _session_expired(r):
        # log in again once instead of failing until the cached session expires
        logger.info("metagraph session expired, logging in again")
        cache.delete(_CACHE_KEY)
        r = get_metagraph_session().get(url, params=params, timeout=settings.MGRAPH_TIMEOUT)
        if _session_expired(r):
            raise RuntimeError("The metagraph service rejected the login")

    if not r.ok:
        body = r.text
        try:
            err = r.json().get("error")
        except ValueError:
            err = None

        # response headers are not logged: they can carry session cookies
        logger.error(
            "metagraph data fetch failed: %s %s\n"
            "Response body (first 500 chars):\n%s\n"
            "Parsed error: %r",
            r.status_code,
            r.reason,
            body[:500],
            err,
        )

        http_err = requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
        raise http_err

    return r.json()
