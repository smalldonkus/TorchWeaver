import unittest
import os
import json
from NNstorage import NNStorage

class TestNNStorage(unittest.TestCase):
    def setUp(self):
        """Set up a fresh test database and load test data."""
        #load in the test file test_nn.json
        self.test_db_path = "test_storage.db"
        self.storage = NNStorage(db_path=self.test_db_path)
        with open("test_nn.json", "r") as f:
            self.test_data = json.load(f)

    def testRemoveDatabase(self):
        """Remove the test database after each test."""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

    def testSaveAndLoad(self):
        """Test saving and loading a neural network."""
        self.storage.save_network("Test Network", self.test_data, "desc")
        networks = self.storage.list_networks()
        self.assertEqual(len(networks), 1)
        net_id = networks[0]["id"]
        loaded = self.storage.load_network(net_id)
        self.assertEqual(loaded, self.test_data)

    def testListNetwork(self):
        """Test listing multiple networks."""
        for i in range(3):
            self.storage.save_network(f"Net {i}", self.test_data, f"desc{i}")
        networks = self.storage.list_networks()
        self.assertEqual(len(networks), 3)
        self.assertEqual(networks[0]["name"], "Net 2")
        self.assertEqual(networks[2]["name"], "Net 0")

    def testUpdateNetwork(self):
        """Test updating a network's data."""
        self.storage.save_network("Net", self.test_data)
        net_id = self.storage.list_networks()[0]["id"]
        updated_data = self.test_data.copy()
        updated_data["nodes"][0]["parameters"]["dims"] = [123]
        self.storage.update_network(net_id, updated_data)
        loaded = self.storage.load_network(net_id)
        self.assertEqual(loaded["nodes"][0]["parameters"]["dims"], [123])

    def testDeleteNetwork(self):
        """Test deleting a network."""
        self.storage.save_network("Net", self.test_data)
        net_id = self.storage.list_networks()[0]["id"]
        self.storage.delete_network(net_id)
        loaded = self.storage.load_network(net_id)
        self.assertIsNone(loaded)

if __name__ == "__main__":
    unittest.main()

