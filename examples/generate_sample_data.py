"""Write the synthetic sample files next to this script (no real people or companies).

The generators live in ``auto_data_scientist.samples`` so the dashboard can offer the same
data without this script; they are re-exported here for the tests and for older commands.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from auto_data_scientist.samples import make_churn, make_orders
except ModuleNotFoundError:  # running from a repository checkout without installation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from auto_data_scientist.samples import make_churn, make_orders

__all__ = ["make_churn", "make_orders"]

if __name__ == "__main__":
    here = Path(__file__).parent
    make_orders().to_csv(here / "orders.csv", index=False)
    make_churn().to_csv(here / "churn.csv", index=False)
    print(f"wrote {here / 'orders.csv'} and {here / 'churn.csv'}")
