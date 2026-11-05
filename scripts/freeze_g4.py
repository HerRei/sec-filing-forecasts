#!/usr/bin/env python3
"""Fill the counts into docs/PREREGISTRATION_G4.md from the frozen dataset and record its hash.

Run once after the dataset build and before `g4 labels`. Refuses if a hash exists already.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disclosure_forecasting.real.g4 import FOLDS, G4, PREREG, PREREG_HASH, folds_table  # noqa: E402

if PREREG_HASH.exists():
    raise SystemExit(f"already frozen: {PREREG_HASH.read_text().strip()}")
build = json.loads((G4 / "build.json").read_text())
folds = folds_table()
names = {
    "dev2019": "2019",
    "dev2020": "2020",
    "dev2021": "2021",
    "dev2022": "2022",
    "final": "2023-01-01 to the last mature decision",
}
rows = []
for fold, (start, end) in FOLDS.items():
    f = folds[fold]
    rows.append(
        f"| {fold} | {names[fold]} ({f['test']:,} events, {f['test_held_out_issuers']:,} from held-out issuers) "
        f"| the 12 months before ({f['val']:,}) | everything earlier ({f['train']:,}) |"
    )
text = PREREG.read_text()
fills = {
    "{EVENTS}": f"{build['events_kept']:,}",
    "{ISSUERS}": f"{build['ciks_kept']}",
    "{FAMILIES}": ", ".join(f"{k}: {v:,}" for k, v in sorted(build["kept_by_family"].items())),
    "{HELD}": f"{build['held_out_issuer_events']:,}",
    "{FOLD_TABLE}": "\n".join(rows),
}
for key, value in fills.items():
    if key not in text:
        raise SystemExit(f"placeholder {key} missing")
    text = text.replace(key, value)
text += (
    "\n## Frozen inputs\n\n"
    + "\n".join(
        f"- `data/processed/g4/{name}`: sha256 `{digest}`"
        for name, digest in build["files"].items()
    )
    + "\n- Counts in the fold table are before the label-availability filter (events without an exit price"
    "\n  are dropped after labelling and counted in `labels.json`).\n"
)
PREREG.write_text(text)
digest = hashlib.sha256(PREREG.read_bytes()).hexdigest()
PREREG_HASH.write_text(f"{digest}  docs/PREREGISTRATION_G4.md\n")
print(digest)
