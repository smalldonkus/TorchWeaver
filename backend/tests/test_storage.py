import unittest
import os
import json
from NNstorage import NNStorage


class TestNNStorage(unittest.TestCase):
    """Test suite for NNstorage database operations"""
    
    def setUp(self):
        """Create a temporary test database"""
        self.test_db_path = "test_storage.db"
        # Remove any existing test database before starting
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except PermissionError:
                pass  # Will be cleaned up later
        self.storage = NNStorage(db_path=self.test_db_path)
        self.test_user_id = "test_user_123"
    
    def tearDown(self):
        """Remove test database after each test"""
        # Close any open connections first
        del self.storage
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except PermissionError:
                pass  # File might still be locked
    
    def test_save_new_network(self):
        """Test saving a new network returns an ID"""
        network_data = {
            "nodes": [{"id": "input1", "type": "input"}],
            "edges": []
        }
        
        network_id = self.storage.save_network(
            name="Test Network",
            json_data=network_data,
            preview_base64="base64data",
            description="Test description",
            user_auth0_id=self.test_user_id
        )
        
        self.assertIsNotNone(network_id)
        self.assertIsInstance(network_id, str)
    
    def test_update_existing_network(self):
        """Test updating an existing network keeps the same ID"""
        network_data = {"nodes": [], "edges": []}
        
        # Save initial network
        network_id = self.storage.save_network(
            name="Original Name",
            json_data=network_data,
            preview_base64="preview1",
            user_auth0_id=self.test_user_id
        )
        
        # Update with same ID
        updated_id = self.storage.save_network(
            name="Updated Name",
            json_data=network_data,
            preview_base64="preview2",
            description="New description",
            network_id=network_id,
            user_auth0_id=self.test_user_id
        )
        
        self.assertEqual(network_id, updated_id)
    
    def test_load_network(self):
        """Test loading a saved network returns correct data"""
        network_data = {
            "nodes": [{"id": "layer1", "type": "linear"}],
            "edges": [{"from": "input", "to": "layer1"}]
        }
        
        network_id = self.storage.save_network(
            name="My Network",
            json_data=network_data,
            preview_base64="preview_data",
            user_auth0_id=self.test_user_id
        )
        
        loaded = self.storage.load_network(network_id, self.test_user_id)
        
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["name"], "My Network")
        self.assertEqual(loaded["preview_base64"], "preview_data")
        self.assertEqual(loaded["nodes"], network_data["nodes"])
        self.assertEqual(loaded["edges"], network_data["edges"])
    
    def test_load_nonexistent_network(self):
        """Test loading a non-existent network returns None"""
        result = self.storage.load_network("fake-id-123", self.test_user_id)
        self.assertIsNone(result)
    
    def test_list_networks(self):
        """Test listing networks returns all saved networks for a user"""
        # Save multiple networks
        self.storage.save_network(
            name="Network 1",
            json_data={"nodes": []},
            preview_base64="preview1",
            user_auth0_id=self.test_user_id
        )
        self.storage.save_network(
            name="Network 2",
            json_data={"nodes": []},
            preview_base64="preview2",
            user_auth0_id=self.test_user_id
        )
        
        networks = self.storage.list_networks(self.test_user_id)
        
        self.assertEqual(len(networks), 2)
        self.assertEqual(networks[0]["name"], "Network 2")  # Most recent first
        self.assertEqual(networks[1]["name"], "Network 1")
    
    def test_list_networks_filtered_by_user(self):
        """Test that list_networks only returns networks for the specified user"""
        # Save network for test user
        self.storage.save_network(
            name="User 1 Network",
            json_data={"nodes": []},
            preview_base64="preview1",
            user_auth0_id=self.test_user_id
        )
        
        # Save network for different user
        self.storage.save_network(
            name="User 2 Network",
            json_data={"nodes": []},
            preview_base64="preview2",
            user_auth0_id="different_user"
        )
        
        networks = self.storage.list_networks(self.test_user_id)
        
        self.assertEqual(len(networks), 1)
        self.assertEqual(networks[0]["name"], "User 1 Network")
    
    def test_delete_network(self):
        """Test deleting a network removes it"""
        network_id = self.storage.save_network(
            name="To Delete",
            json_data={"nodes": []},
            preview_base64="preview",
            user_auth0_id=self.test_user_id
        )
        
        result = self.storage.delete_network(network_id, self.test_user_id)
        
        self.assertTrue(result)
        loaded = self.storage.load_network(network_id, self.test_user_id)
        self.assertIsNone(loaded)
    
    def test_delete_network_wrong_user(self):
        """Test that deleting with wrong user ID doesn't delete the network"""
        network_id = self.storage.save_network(
            name="Protected Network",
            json_data={"nodes": []},
            preview_base64="preview",
            user_auth0_id=self.test_user_id
        )
        
        self.storage.delete_network(network_id, "wrong_user")
        
        # Network should still exist
        loaded = self.storage.load_network(network_id, self.test_user_id)
        self.assertIsNotNone(loaded)
    
    def test_update_network_json(self):
        """Test updating network JSON data"""
        network_id = self.storage.save_network(
            name="Network",
            json_data={"nodes": [{"id": "1"}]},
            preview_base64="preview",
            user_auth0_id=self.test_user_id
        )
        
        new_data = {"nodes": [{"id": "1"}, {"id": "2"}]}
        result = self.storage.update_network(network_id, new_data, self.test_user_id)
        
        self.assertTrue(result)
        loaded = self.storage.load_network(network_id, self.test_user_id)
        self.assertEqual(len(loaded["nodes"]), 2)
    
    def test_set_favourite_status_true(self):
        """Test setting network as favourite"""
        network_id = self.storage.save_network(
            name="Favourite Network",
            json_data={"nodes": []},
            preview_base64="preview",
            user_auth0_id=self.test_user_id
        )
        
        rows_updated = self.storage.set_favourite_status(network_id, True)
        
        self.assertEqual(rows_updated, 1)
        networks = self.storage.list_networks(self.test_user_id)
        self.assertEqual(networks[0]["favourited"], 1)
    
    def test_set_favourite_status_false(self):
        """Test removing favourite status"""
        network_id = self.storage.save_network(
            name="Network",
            json_data={"nodes": []},
            preview_base64="preview",
            user_auth0_id=self.test_user_id
        )
        
        # Set as favourite
        self.storage.set_favourite_status(network_id, True)
        # Remove favourite
        rows_updated = self.storage.set_favourite_status(network_id, False)
        
        self.assertEqual(rows_updated, 1)
        networks = self.storage.list_networks(self.test_user_id)
        self.assertEqual(networks[0]["favourited"], 0)
    
    def test_database_table_creation(self):
        """Test that database table is created on initialization"""
        # The table should already be created in setUp
        # Just verify we can perform operations without errors
        network_id = self.storage.save_network(
            name="Test",
            json_data={"nodes": []},
            preview_base64="preview",
            user_auth0_id=self.test_user_id
        )
        self.assertIsNotNone(network_id)


if __name__ == '__main__':
    unittest.main()
