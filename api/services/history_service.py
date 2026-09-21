"""Persistent reconstruction history service using SQLite."""

import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from api.config import settings
from api.schemas.history import ReconstructionHistoryItem

logger = logging.getLogger(__name__)


class HistoryService:
    """Manages persistent reconstruction history using a local SQLite database."""

    def __init__(self, db_path: Optional[str] = None):
        raw_path = db_path or settings.HISTORY_DB_PATH
        path_obj = Path(raw_path)
        if not path_obj.is_absolute():
            repo_root = Path(__file__).resolve().parent.parent.parent
            self.db_path = repo_root / raw_path
        else:
            self.db_path = path_obj

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database table and index if they do not exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS reconstruction_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        latitude REAL NOT NULL,
                        longitude REAL NOT NULL,
                        regime TEXT NOT NULL,
                        timestamp TEXT NOT NULL
                    );
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_reconstruction_history_id
                    ON reconstruction_history(id DESC);
                    """
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error initializing reconstruction history database: {e}")

    def save_reconstruction(
        self,
        date: str,
        latitude: float,
        longitude: float,
        regime: str,
        timestamp: Optional[str] = None,
    ) -> ReconstructionHistoryItem:
        """Save a successful reconstruction run to the persistent history."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO reconstruction_history (date, latitude, longitude, regime, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (date, float(latitude), float(longitude), regime, ts),
            )
            item_id = cursor.lastrowid
            conn.commit()

        return ReconstructionHistoryItem(
            id=item_id,
            date=date,
            latitude=latitude,
            longitude=longitude,
            regime=regime,
            timestamp=ts,
        )

    def get_recent_history(self, limit: int = 50) -> List[ReconstructionHistoryItem]:
        """Retrieve recent reconstruction history records, ordered newest first."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, date, latitude, longitude, regime, timestamp
                FROM reconstruction_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()

        return [
            ReconstructionHistoryItem(
                id=row["id"],
                date=row["date"],
                latitude=row["latitude"],
                longitude=row["longitude"],
                regime=row["regime"],
                timestamp=row["timestamp"],
            )
            for row in rows
        ]

    def clear_history(self) -> None:
        """Clear all records from the history table (useful for testing)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM reconstruction_history;")
            conn.commit()


_history_service_instance: Optional[HistoryService] = None


def get_history_service() -> HistoryService:
    """Dependency provider returning singleton HistoryService."""
    global _history_service_instance
    if _history_service_instance is None:
        _history_service_instance = HistoryService()
    return _history_service_instance
