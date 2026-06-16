#!/usr/bin/env python3
"""One-time helper: merge legacy split stores into config/{client}.json."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def merge_client(client_id: str) -> Path | None:
    website = ROOT / "data" / "clients" / client_id / "knowledge_website.json"
    pdfs = ROOT / "data" / "clients" / client_id / "knowledge_pdfs.json"
    legacy = ROOT / "data" / "clients" / client_id / "knowledge_base.json"
    output = ROOT / "config" / f"{client_id}.json"

    documents: list[dict] = []
    for path in (legacy, website, pdfs):
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            documents.extend(payload)
        elif isinstance(payload, dict):
            documents.extend(payload.get("documents", []))

    if not documents:
        return None

    output.write_text(
        json.dumps({"client_id": client_id, "documents": documents}, indent=2),
        encoding="utf-8",
    )
    return output


def main() -> int:
    written = 0
    for client_id in ("client-1", "client-2"):
        path = merge_client(client_id)
        if path:
            print(f"Wrote {path}")
            written += 1
        else:
            print(f"No legacy data found for {client_id}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
