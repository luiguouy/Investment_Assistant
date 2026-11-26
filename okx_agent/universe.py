"""OKX trading universe and per-instrument policy loader."""

import json
from pathlib import Path

from risk_ledger import get_bucket

DEFAULT_UNIVERSE_PATH = Path(__file__).resolve().parent.parent / "config" / "okx_universe.json"


def load_universe(universe_path: str | Path = DEFAULT_UNIVERSE_PATH) -> dict:
    """Load the OKX instrument universe configuration."""
    path = Path(universe_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_instrument_policy(
    inst_id: str,
    universe_path: str | Path = DEFAULT_UNIVERSE_PATH,
) -> dict:
    """Return merged defaults, perp-type rules, and instrument metadata for inst_id."""
    data = load_universe(universe_path)
    sym = inst_id.strip().upper()
    defaults = dict(data.get("defaults", {}))
    perp_rules = data.get("perp_type_rules", {})
    instruments = data.get("instruments", {})

    inst_cfg = instruments.get(sym)
    in_universe = inst_cfg is not None
    if inst_cfg is None:
        perp_type = "crypto"
        enabled = False
        bucket = get_bucket(sym)
    else:
        perp_type = inst_cfg.get("perp_type", "crypto")
        enabled = bool(inst_cfg.get("enabled", True))
        bucket = inst_cfg.get("bucket") or get_bucket(sym)

    type_rule = dict(perp_rules.get(perp_type, perp_rules.get("crypto", {})))
    return {
        "inst_id": sym,
        "in_universe": in_universe,
        "enabled": enabled,
        "perp_type": perp_type,
        "bucket": bucket,
        "defaults": defaults,
        "perp_type_rules": type_rule,
        "min_liq_distance_r": float(type_rule.get("min_liq_distance_r", 2.0)),
    }
