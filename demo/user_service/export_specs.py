"""Export user-service OpenAPI specs into fixtures/.

    python demo/user_service/export_specs.py          # (re)write the fixtures
    python demo/user_service/export_specs.py --check  # exit 1 if fixtures are stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from user_service.app import create_app
from user_service.contracts import CONTRACTS

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"

# Fixture file -> contract version. All scenarios compare against openapi_before.json.
OUTPUTS = {
    "openapi_before.json": "v1",
    "openapi_after.json": "user_id_renamed",  # scenario 1, also the README example
    "openapi_after_email_renamed.json": "email_renamed",
    "openapi_after_phone_added.json": "phone_added",
    "openapi_after_newsletter_removed.json": "newsletter_removed",
    "openapi_after_locale_removed.json": "locale_removed",
}


def render(contract_name: str) -> str:
    return json.dumps(create_app(CONTRACTS[contract_name]).openapi(), indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify fixtures instead of writing them")
    args = parser.parse_args()

    stale = []
    for filename, contract_name in OUTPUTS.items():
        path = FIXTURES / filename
        expected = render(contract_name)
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                stale.append(filename)
        else:
            path.write_text(expected, encoding="utf-8")
            print(f"wrote {path.relative_to(FIXTURES.parent)}")
    if stale:
        print(f"stale fixtures (run without --check to regenerate): {', '.join(stale)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
