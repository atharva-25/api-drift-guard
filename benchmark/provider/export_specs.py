"""Export benchmark provider specs into benchmark/specs/.

    python benchmark/provider/export_specs.py          # (re)write
    python benchmark/provider/export_specs.py --check  # exit 1 if stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bench_user_service.app import CONTRACTS, create_app

SPECS = Path(__file__).resolve().parents[1] / "specs"


def render(name: str) -> str:
    return json.dumps(create_app(CONTRACTS[name]).openapi(), indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    SPECS.mkdir(exist_ok=True)
    stale = []
    for name in CONTRACTS:
        path = SPECS / f"{name}.json"
        expected = render(name)
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                stale.append(path.name)
        else:
            path.write_text(expected, encoding="utf-8")
            print(f"wrote {path.relative_to(SPECS.parents[1])}")
    if stale:
        print(f"stale specs: {', '.join(stale)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
