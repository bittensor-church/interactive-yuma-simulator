import logging
from urllib.parse import urljoin
from datetime import datetime

import requests
import pandas as pd
from django.conf import settings

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


METAGRAPH_DATA_PATH = "api/v1/metagraph-data/"


def fetch_metagraph_data(
    start_date: datetime,
    end_date: datetime,
    netuid: int,
) -> dict:
    """
    Fetch per-epoch metagraph snapshots (hotkeys, stakes, weights, labels) from the metagraph
    service. The service authenticates callers with a bearer token.
    """
    if not settings.MGRAPH_BASE_URL or not settings.MGRAPH_API_TOKEN:
        raise RuntimeError("The metagraph service is not configured (MGRAPH_BASE_URL, MGRAPH_API_TOKEN).")

    url = urljoin(settings.MGRAPH_BASE_URL.rstrip("/") + "/", METAGRAPH_DATA_PATH)
    params = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "netuid": netuid,
    }

    logger.debug("→ GET %s %r", url, params)
    r = requests.get(
        url,
        params=params,
        headers={"Authorization": f"Bearer {settings.MGRAPH_API_TOKEN}"},
        timeout=settings.MGRAPH_TIMEOUT,
    )

    if not r.ok:
        body = r.text
        try:
            err = r.json().get("error")
        except ValueError:
            err = None

        # response headers are not logged
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
