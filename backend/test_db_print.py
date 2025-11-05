import sqlite3
import json
import os

DB_PATH = "NN_storage.db"  # adjust path if needed (e.g., "backend/NN_storage.db")

def view_all_networks():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Show all networks
    cur.execute("SELECT id, name, description, created_at, json_data FROM networks")
    rows = cur.fetchall()

    if not rows:
        print(" No networks found in the database.")
        return

    print(f" Found {len(rows)} network(s):\n")

    for r in rows:
        network_id, name, desc, created_at, json_str = r
        print("=" * 60)
        print(f"ID: {network_id}")
        print(f"Name: {name}")
        print(f"Description: {desc}")
        print(f"Created At: {created_at}")

        try:
            data = json.loads(json_str)
            print(f"Nodes: {len(data.get('nodes', []))}")
            print(f"Edges: {len(data.get('edges', []))}")

            print("Full node data:")
            print(json.dumps(data.get("nodes", []), indent=2))

            print("Full edge data:")
            print(json.dumps(data.get("edges", []), indent=2))

        except Exception as e:
            print("Could not parse JSON data:", e)

        print("=" * 60, "\n")

    conn.close()

if __name__ == "__main__":
    view_all_networks()
