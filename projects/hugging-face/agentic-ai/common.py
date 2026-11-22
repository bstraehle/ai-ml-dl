import os
from datetime import date

MODEL = "gpt-6-luna"
REASONING_EFFORT = "low"
TIMEOUT = 60


def today() -> str:
    """Returns today's date. Use this for any questions related to knowing today's date.
    Any date mathematics should occur outside this function."""
    return str(date.today())


def openweathermap_api_key() -> str:
    key = os.environ.get("OPENWEATHERMAP_API_KEY")
    if not key:
        raise RuntimeError("OPENWEATHERMAP_API_KEY is not set.")
    return key
