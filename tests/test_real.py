"""Real-data stages: timing convention, label gate, purged folds, timestamp classification."""

import numpy as np
import pandas as pd
import pytest

from disclosure_forecasting.real import g4, sessions


def executable(utc):
    out = sessions.executable_from_acceptance(pd.Series([utc]))
    return pd.Timestamp(out["executable_at"].iloc[0]).strftime("%Y-%m-%d %H:%M")


def test_pre_market_filing_trades_at_the_same_open():
    # CAT's Q2 2024 earnings 8-K: accepted 06:32:05 EDT = 10:32:05 UTC -> open 09:30 EDT = 13:30 UTC
    assert executable("2024-08-06T10:32:05Z") == "2024-08-06 13:30"


def test_filing_inside_the_buffer_or_session_waits_for_the_next_open():
    assert executable("2024-08-07T13:10:00Z") == "2024-08-08 13:30"  # 09:10 EDT + 30 min > 09:30
    assert executable("2024-08-07T14:40:25Z") == "2024-08-08 13:30"  # 10:40 EDT, during the session
    assert executable("2024-08-07T20:05:00Z") == "2024-08-08 13:30"  # after the close


def test_holidays_weekends_and_winter_time():
    # Friday 14:36 EST before Presidents' Day: next open is Tuesday 09:30 EST = 14:30 UTC
    assert executable("2025-02-14T19:36:30Z") == "2025-02-18 14:30"
    # Saturday acceptance -> Monday open
    assert executable("2024-08-10T12:00:00Z") == "2024-08-12 13:30"


def test_day_level_fallback_never_uses_the_same_day():
    out = sessions.executable_from_filing_date(pd.Series(["2024-08-06"]))
    assert pd.Timestamp(out["exec_session"].iloc[0]) == pd.Timestamp("2024-08-07")


def test_labels_refuse_without_recorded_preregistration(tmp_path, monkeypatch):
    monkeypatch.setattr(g4, "PREREG", tmp_path / "missing.md")
    monkeypatch.setattr(g4, "PREREG_HASH", tmp_path / "missing.sha256")
    with pytest.raises(SystemExit):
        g4.labels()
    (tmp_path / "missing.md").write_text("plan")
    (tmp_path / "missing.sha256").write_text("0" * 64)
    with pytest.raises(SystemExit):  # hash does not match the file
        g4.labels()


def test_folds_purge_label_windows_and_hold_out_issuers(monkeypatch):
    days = sessions.session_index()
    start = int(np.searchsorted(days.values, np.datetime64("2018-01-02")))
    s = np.arange(start, start + 1500, 3)
    events = pd.DataFrame({"s": s, "held_out_issuer": (s % 5 == 0)})
    monkeypatch.setattr(g4, "last_label_session", lambda: start + 1500 + 40)
    for fold, (test_start, _) in g4.FOLDS.items():
        train, val, test = g4.fold_masks(events, fold)
        first_test = int(np.searchsorted(days.values, np.datetime64(test_start)))
        first_val = int(
            np.searchsorted(days.values, np.datetime64(test_start) - np.timedelta64(365, "D"))
        )
        gap = g4.LABEL_SESSIONS + g4.EMBARGO
        assert (s[train] + gap <= first_val).all()
        assert (s[val] + gap <= first_test).all()
        assert not (train & val).any() and not (val & test).any() and not (train & test).any()
        assert not events["held_out_issuer"].values[train | val].any()
