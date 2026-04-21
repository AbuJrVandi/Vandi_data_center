from __future__ import annotations

import json
from pathlib import Path

from .models import OperationRecord


class OperationLogger:
    def __init__(self, log_path: str | Path = "logs/operations.jsonl") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[OperationRecord] = []

    def record(self, record: OperationRecord) -> None:
        self._records.append(record)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), default=str) + "\n")

    def list_records(self) -> list[OperationRecord]:
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()
        if self.log_path.exists():
            self.log_path.unlink()
