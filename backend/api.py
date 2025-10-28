from flask import Flask, request, jsonify
from flask_cors import CORS
from NNgenerator import generate
from NNdatabase import NNDataBase
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the database to load operation definitions
try:
    db = NNDataBase()
except Exception as e:
    print(f"Warning: Could not initialize NNDataBase: {e}")
    db = None

@app.route('/convert-json-to-python', methods=['POST'])
def convert_json_to_python():
    try:
        # Get JSON data from the request
        json_data = request.get_json()
        
        if not json_data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Get tensor operations configuration from database
        tensor_ops_config = []
        if db and db.defaults.get("tensorOp"):
            tensor_ops_config = db.defaults["tensorOp"].get("data", [])
        
        # Check if we have the new format (nodes) or old format (layers)
        if "nodes" in json_data:
            # Use new nodes format directly
            python_code = generate(json_data, tensor_ops_config)
        elif "layers" in json_data:
            # Convert old layers format to new nodes format
            converted_data = {
                "version": json_data.get("version", 1.0),
                "libraries": json_data.get("libraries", {}),
                "inputs": json_data.get("inputs", []),
                "nodes": json_data["layers"],  # Convert layers to nodes
                "outputs": json_data.get("outputs", [])
            }
            python_code = generate(converted_data, tensor_ops_config)
        else:
            return jsonify({"error": "Missing 'nodes' or 'layers' field in JSON"}), 400
        
        return jsonify({
            "success": True,
            "python_code": python_code
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/api/operations/layers', methods=['GET'])
def get_layers():
    """Get available layer definitions"""
    try:
        if db and db.defaults.get("nnLayer"):
            return jsonify(db.defaults["nnLayer"]), 200
        else:
            return jsonify({"error": "Layer definitions not available"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/operations/tensorops', methods=['GET'])
def get_tensor_ops():
    """Get available tensor operation definitions"""
    try:
        if db and db.defaults.get("tensorOp"):
            return jsonify(db.defaults["tensorOp"]), 200
        else:
            return jsonify({"error": "Tensor operation definitions not available"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/operations/activators', methods=['GET'])
def get_activators():
    """Get available activator definitions"""
    try:
        if db and db.defaults.get("activator"):
            return jsonify(db.defaults["activator"]), 200
        else:
            return jsonify({"error": "Activator definitions not available"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/operations/all', methods=['GET'])
def get_all_operations():
    """Get all operation definitions in one call"""
    try:
        if not db:
            return jsonify({"error": "Database not initialized"}), 500
            
        return jsonify({
            "layers": db.defaults.get("nnLayer", {"data": []}),
            "tensorOps": db.defaults.get("tensorOp", {"data": []}),
            "activators": db.defaults.get("activator", {"data": []})
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/variables-info', methods=['GET'])
def get_variables_info():
    """Get variable type information"""
    try:
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        variables_info_path = os.path.join(script_dir, "JsonLayers", "variables_info.json")
        
        with open(variables_info_path, "r") as f:
            data = json.load(f)
            return jsonify(data), 200
    except FileNotFoundError:
        return jsonify({"error": "variables_info.json not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)