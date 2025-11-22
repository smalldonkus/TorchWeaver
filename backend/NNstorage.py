import sqlite3
import json
from datetime import datetime
import uuid

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
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TEXT,
                updated_at TEXT,
                json_data TEXT NOT NULL,
                user_auth0_id TEXT,
                favourited BOOLEAN DEFAULT 0,
                preview_base64 TEXT
            )
        """)
        conn.commit()
        conn.close()

    def save_network(self, name, json_data, preview_base64, description=None, network_id=None, user_auth0_id=None):
        """
        Save or update a neural network structure with metadata.

        If `network_id` is provided and exists in the database,
        the record is updated instead of inserted.
        """
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Check if we’re updating an existing record
        if network_id is not None:
            cur.execute("SELECT id FROM networks WHERE id = ?", (network_id,))
            existing = cur.fetchone()
            if existing:
                # Update existing record
                cur.execute("""
                    UPDATE networks
                    SET name = ?, description = ?, updated_at = ?, json_data = ?, user_auth0_id = ?, preview_base64 = ?
                    WHERE id = ?
                """, (name, description, datetime.now().isoformat(), json.dumps(json_data), user_auth0_id, preview_base64, network_id))
                conn.commit()
                conn.close()
                print(f"[DB] Updated network ID={network_id}")
                return network_id

        # Otherwise, insert a new record
        new_id = str(uuid.uuid4())
        cur.execute("""
            INSERT INTO networks (id, name, description, created_at, json_data, user_auth0_id, preview_base64)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (new_id, name, description, datetime.now().isoformat(), json.dumps(json_data), user_auth0_id, preview_base64))
        conn.commit()
        conn.close()
        print(f"[DB] Saved new network ID={new_id} for user={user_auth0_id}")
        return new_id

    def list_networks(self, user_auth0_id: str):
        """Return metadata for all saved networks for a given user."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, name, description, created_at, favourited, preview_base64 FROM networks WHERE user_auth0_id = ? ORDER BY id DESC, created_at DESC", (user_auth0_id,))
        rows = cur.fetchall()
        conn.close()
        return [
            {"id": r[0], "name": r[1], "description": r[2], "created_at": r[3], "favourited": r[4], "preview_base64": r[5]}
            for r in rows
        ]

    def load_network(self, network_id, user_auth0_id: str = None):
        """Load a specific network by ID for a given user (if provided)."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        if user_auth0_id:
            cur.execute("SELECT name, preview_base64, json_data FROM networks WHERE id = ? AND user_auth0_id = ?", (network_id, user_auth0_id))
        else:
            cur.execute("SELECT name, preview_base64, json_data FROM networks WHERE id = ?", (network_id,))
        
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        name, preview_base64, json_data = row
        network = json.loads(json_data)
        network["name"] = name
        network["preview_base64"] = preview_base64
        return network

    def delete_network(self, network_id, user_auth0_id: str = None):
        """Delete a saved network. If user_auth0_id is given, only delete if it matches."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        if user_auth0_id:
            cur.execute("DELETE FROM networks WHERE id = ? AND user_auth0_id = ?", (network_id, user_auth0_id))
        else:
            cur.execute("DELETE FROM networks WHERE id = ?", (network_id,))
        conn.commit()
        conn.close()
        return True

    def update_network(self, network_id, json_data, user_auth0_id: str = None):
        """Update an existing network's JSON data. If user_auth0_id provided, only update matching record."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        if user_auth0_id:
            cur.execute("""
                UPDATE networks
                SET json_data = ?, updated_at = ?
                WHERE id = ? AND user_auth0_id = ?
            """, (json.dumps(json_data), datetime.now().isoformat(), network_id, user_auth0_id))
        else:
            cur.execute("""
                UPDATE networks
                SET json_data = ?, updated_at = ?
                WHERE id = ?
            """, (json.dumps(json_data), datetime.now().isoformat(), network_id))
        conn.commit()
        conn.close()
        return True
    
    def set_favourite_status(self, network_id, favourited):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            UPDATE networks
            SET favourited = ?
            WHERE id = ?
        """, (favourited, network_id))
        conn.commit()
        rows_updated = cur.rowcount
        conn.close()
        return rows_updated