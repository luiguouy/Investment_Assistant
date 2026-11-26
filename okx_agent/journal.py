"""Append-only structured trade & decision audit journal (JSONL)."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_JOURNAL_PATH = Path(__file__).resolve().parent.parent / "data" / "okx_journal.jsonl"


def append_journal_entry(
    event_type: str,
    inst_id: str,
    payload: dict[str, Any],
    journal_path: str | Path = DEFAULT_JOURNAL_PATH,
) -> dict[str, Any]:
    """Append a timestamped decision or execution record to okx_journal.jsonl."""
    path = Path(journal_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type.strip().upper(),
        "inst_id": inst_id.strip().upper(),
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def read_journal(
    journal_path: str | Path = DEFAULT_JOURNAL_PATH,
    inst_id: str | None = None,
    event_type: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Read and optionally filter records from okx_journal.jsonl."""
    path = Path(journal_path)
    if not path.exists():
        return []
    inst_filter = inst_id.strip().upper() if inst_id else None
    type_filter = event_type.strip().upper() if event_type else None
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if inst_filter and item.get("inst_id") != inst_filter:
                continue
            if type_filter and item.get("event_type") != type_filter:
                continue
            rows.append(item)
    if limit is not None and limit > 0:
        return rows[-limit:]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="OKX Agent Trade & Decision Audit Journal")
    parser.add_argument("--journal", default=str(DEFAULT_JOURNAL_PATH))
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_app = sub.add_parser("append")
    p_app.add_argument("--event-type", required=True)
    p_app.add_argument("--inst-id", required=True)
    p_app.add_argument("--payload", required=True, help="JSON object string")

    p_tail = sub.add_parser("tail")
    p_tail.add_argument("--inst-id", default=None)
    p_tail.add_argument("--event-type", default=None)
    p_tail.add_argument("--limit", type=int, default=20)

    args = parser.parse_args()
    if args.cmd == "append":
        rec = append_journal_entry(
            args.event_type,
            args.inst_id,
            json.loads(args.payload),
            journal_path=args.journal,
        )
        print(json.dumps(rec, indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "tail":
        rows = read_journal(
            journal_path=args.journal,
            inst_id=args.inst_id,
            event_type=args.event_type,
            limit=args.limit,
        )
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
