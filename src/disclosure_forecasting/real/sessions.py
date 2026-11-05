"""NYSE sessions (exchange_calendars XNYS) and the conservative executable_at convention.

Convention (fixed before any outcome was computed):
- public_at = EDGAR acceptance time (exact to the second, UTC in the submissions JSON; checked
  against the Eastern-time "Accepted" line of filing index pages in the Gate 1 audit).
- executable_at = the first regular-session OPEN at or after public_at + 30 minutes.
  A filing accepted at 06:32 ET is executable at that day's 09:30 open; one accepted at 09:10 ET
  or during the session waits for the next day's open; one accepted after the close too.
- If a filing has no acceptance time, only its filing date D is known. EDGAR assigns D to anything
  accepted up to 22:00 ET on D, so executable_at = the open of the first session after D.
"""

from functools import lru_cache

import exchange_calendars as xcals
import numpy as np
import pandas as pd

BUFFER = pd.Timedelta(minutes=30)
START, END = "2005-01-03", "2026-12-31"


@lru_cache(maxsize=1)
def calendar():
    return xcals.get_calendar("XNYS", start=START, end=END)


@lru_cache(maxsize=1)
def opens() -> pd.Series:
    """Session date (tz-naive midnight) -> open time (UTC)."""
    cal = calendar()
    return cal.schedule["open"]


def session_index() -> pd.DatetimeIndex:
    return opens().index


def executable_from_acceptance(accepted_utc: pd.Series) -> pd.DataFrame:
    """Vectorised: first session open >= accepted + 30 min. Returns session date and open time."""
    o = opens()
    open_times = o.values.astype("datetime64[ns]")
    ts = pd.to_datetime(accepted_utc, utc=True).dt.tz_convert(None) + BUFFER
    position = np.searchsorted(open_times, ts.values.astype("datetime64[ns]"), side="left")
    valid = position < len(o)
    position = np.where(valid, position, len(o) - 1)
    return pd.DataFrame(
        {
            "exec_session": np.where(valid, o.index.values[position], np.datetime64("NaT")),
            "executable_at": np.where(valid, open_times[position], np.datetime64("NaT")),
        },
        index=accepted_utc.index,
    )


def executable_from_filing_date(filing_date: pd.Series) -> pd.DataFrame:
    """Day-level fallback: the open of the first session strictly after the filing date."""
    o = opens()
    sessions = o.index.values.astype("datetime64[ns]")
    days = pd.to_datetime(filing_date).values.astype("datetime64[ns]")
    position = np.searchsorted(sessions, days, side="right")
    valid = position < len(o)
    position = np.where(valid, position, len(o) - 1)
    return pd.DataFrame(
        {
            "exec_session": np.where(valid, o.index.values[position], np.datetime64("NaT")),
            "executable_at": np.where(
                valid, o.values.astype("datetime64[ns]")[position], np.datetime64("NaT")
            ),
        },
        index=filing_date.index,
    )


def eastern(ts) -> pd.Series:
    return pd.to_datetime(ts, utc=True).dt.tz_convert("America/New_York")
