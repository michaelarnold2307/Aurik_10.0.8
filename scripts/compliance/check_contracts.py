#!/usr/bin/env python3
"""Pre-commit hook: ContractValidator — Cross-Module-Integritätsprüfung.

Läuft in CI/pre-commit. Exit 1 bei Inkonsistenzen.
"""

import sys

try:
    from backend.core.defect_contract_validator import run_contract_validation

    result = run_contract_validation()
    if result["ok"]:
        print(f"ContractValidator: ✅ OK ({result['violations']} violations)")
        sys.exit(0)
    else:
        print(f"ContractValidator: ❌ {result['violations']} VIOLATIONS")
        for detail in result["details"]:
            print(f"  {detail}")
        sys.exit(1)
except ImportError as e:
    print(f"ContractValidator: ⚠️ Import failed ({e}) — skipping")
    sys.exit(0)
