from flask import Blueprint, request, jsonify
from flask_cors import CORS
from NNstorage import NNStorage
import traceback

NNRoutes  = Blueprint("nn_routes", __name__)
CORS(NNRoutes)
storage = NNStorage()

# current implementation get id from header or payload
# however there are security concerns with this approach
# this should be addressed in future iterations
# saving require user id to be passed in header or payload
# others can be use db generated id
def get_user_id(req):
    user_id = None

    try:
        data = req.get_json(silent=True) or {}
        if isinstance(data, dict):
            user = data.get('user')
            if isinstance(user, dict):
                user_id = user.get('id')
    except Exception:
        user_id = None

    # now swapped the logic to check db first then header
    if not user_id and 'header' in req.headers:
        user_id = req.headers.get('header')

    return user_id


@NNRoutes .route('/save_network', methods=['POST'])
def saveNetwork():
    try:
        print("Save_network called")
        data = request.get_json()

        print("Printing payload:", data)

        name = data.get("name")
        network_id = data.get("nn_id")
        preview_base64 = data.get("preview")
        # Prefer authenticated user id from headers, fall back to payload
        user_auth0_id = get_user_id(request)
        if not user_auth0_id:
            user_info = data.get("user")
            user_auth0_id = user_info.get("id") if user_info else None
        # Ensure network_id is int or None
        if network_id is not None:
            try:
                network_id = int(network_id)
            except Exception:
                network_id = None

        json_data = data.get("network")
        description = data.get("description", None)

        if not name or not json_data:
            return jsonify({"error": "Missing 'name' or 'network' data"}), 400

        # Save network and get the ID back (storage will insert or update based on network_id)
        saved_id = storage.save_network(name, json_data, preview_base64, description, network_id, user_auth0_id)
        return jsonify({"success": True, "id": saved_id}), 200
    except Exception as e:
        print("[NNRoutes] Exception in save_network:")
        traceback.print_exc()
        return jsonify({ "error": str(e)}), 500
    
@NNRoutes .route('/list_network', methods=['GET'])
def listNetwork():
    try:
        # Extract user id from request (header or payload). Frontend should send header header or include user info in body.
        user_id = get_user_id(request)
        if not user_id:
            return jsonify({"error": "Unauthorized - missing user id"}), 401

        networks = storage.list_networks(user_id)
        return jsonify({"networks": networks}), 200
    except Exception as e:
        return jsonify({ "error": str(e)}), 500

@NNRoutes .route('/load_network', methods=['GET'])
def loadNetwork():
    try:
        network_id = request.args.get("id")
        if not network_id:
            return jsonify({"error": "Missing 'id' parameter"}), 400
        user_id = get_user_id(request)
        network = storage.load_network(network_id, user_id)
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
        user_id = get_user_id(request)
        if not user_id:
            return jsonify({"error": "Unauthorized - missing user id"}), 401
        storage.delete_network(network_id, user_id)
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
        user_id = get_user_id(request)
        storage.update_network(network_id, json_data, user_id)
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({ "error": str(e)}), 500
    
@NNRoutes.route('/favourite_network', methods=['POST'])
def favourite_network():
    try:
        data = request.get_json()
        network_id = data.get("id")
        favourited = data.get("favourited")

        if not network_id:
            return jsonify({"error": "Missing 'network_id'"}), 400

        storage.set_favourite_status(network_id, favourited)
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500