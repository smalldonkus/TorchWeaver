import sqlite3
import json
from datetime import datetime

class NNStorage:
    def __init__(self, db_path="NN_storage.db"):
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        """Create the database table if it doesn’t exist."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS networks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                created_at TEXT,
                json_data TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def save_network(self, name, json_data, description=None):
        """Save a neural network structure with metadata."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO networks (name, description, created_at, json_data)
            VALUES (?, ?, ?, ?)
        """, (name, description, datetime.now().isoformat(), json.dumps(json_data)))
        conn.commit()
        conn.close()
        return True

    def list_networks(self):
        """Return metadata for all saved networks."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, name, description, created_at FROM networks ORDER BY id DESC")
        rows = cur.fetchall()
        conn.close()
        return [
            {"id": r[0], "name": r[1], "description": r[2], "created_at": r[3]}
            for r in rows
        ]

    def load_network(self, network_id):
        """Load a specific network by ID."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT json_data FROM networks WHERE id = ?", (network_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return json.loads(row[0])

    def delete_network(self, network_id):
        """Delete a saved network."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("DELETE FROM networks WHERE id = ?", (network_id,))
        conn.commit()
        conn.close()
        return True

    def update_network(self, network_id, json_data):
        """Update an existing network's JSON data."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            UPDATE networks
            SET json_data = ?, created_at = ?
            WHERE id = ?
        """, (json.dumps(json_data), datetime.now().isoformat(), network_id))
        conn.commit()
        conn.close()
        return True