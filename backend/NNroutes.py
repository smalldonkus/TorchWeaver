from flask import Blueprint, request, jsonify
from flask_cors import CORS
from NNstorage import NNStorage
import traceback

NNRoutes  = Blueprint("nn_routes", __name__)
CORS(NNRoutes)
storage = NNStorage()

@NNRoutes .route('/save_network', methods=['POST'])
def saveNetwork():
    try:
        print("Save_network called")
        data = request.get_json()

        print("Printing payload:", data)

        name = data.get("name")
        json_data = data.get("network")
        description = data.get("description", None)

        if not name or not json_data:
            return jsonify({"error": "Missing 'name' or 'network' data"}), 400

        # storage.save_network expects (name, json_data, description=None)
        storage.save_network(name, json_data, description)
        return jsonify({"success": True}), 200
    except Exception as e:
        print("[NNRoutes] Exception in save_network:")
        traceback.print_exc()
        return jsonify({ "error": str(e)}), 500
    
@NNRoutes .route('/list_network', methods=['GET'])
def listNetwork():
    try:
        networks = storage.list_networks()
        return jsonify({"networks": networks}), 200
    except Exception as e:
        return jsonify({ "error": str(e)}), 500

@NNRoutes .route('/load_network', methods=['GET'])
def loadNetwork():
    try:
        network_id = request.args.get("id")
        if not network_id:
            return jsonify({"error": "Missing 'id' parameter"}), 400

        network = storage.load_network(network_id)
        if not network:
            return jsonify({"error": "Network not found"}), 404
        
        print("Loading network:", network)  # Debug log
        # Ensure we have the correct structure
        if not isinstance(network, dict) or 'nodes' not in network or 'edges' not in network:
            print("Warning: Network data not in expected format")
            # If network is direct nodes/edges array, wrap it
            return jsonify({"network": {
                "nodes": network.get("nodes", []),
                "edges": network.get("edges", [])
            }}), 200

        return jsonify({"network": network}), 200
    except Exception as e:
        return jsonify({ "error": str(e)}), 500

@NNRoutes .route('/delete_network', methods=['DELETE'])
def deleteNetwork():
    try:
        network_id = request.args.get("id")
        if not network_id:
            return jsonify({"error": "Missing 'id' parameter"}), 400

        storage.delete_network(network_id)
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({ "error": str(e)}), 500

@NNRoutes .route('/update_network', methods=['PUT'])
def updateNetwork():
    try:
        data = request.get_json()
        network_id = data.get("id")
        json_data = data.get("json_data")

        if not network_id or not json_data:
            return jsonify({"error": "Missing 'id' or 'json_data'"}), 400

        storage.update_network(network_id, json_data)
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({ "error": str(e)}), 500