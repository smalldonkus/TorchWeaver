from flask import Flask, request, jsonify
from flask_cors import CORS
from parse import parse
from NNgenerator import generate
from NNdatabase import NNDataBase
from NNroutes import NNRoutes
import json
import traceback


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

app.register_blueprint(NNRoutes)

# Initialize the database to load operation definitions
try:
    db = NNDataBase()
except Exception as e:
    print(f"Warning: Could not initialize NNDataBase: {e}")
    db = None

# SIMPLIFIED API STRUCTURE
# /api/operations/* - For getting operation definitions
# /api/generate - For code generation
# /api/parse - For validation/parsing
# /health - For health checks

@app.route('/api/generate', methods=['POST'])
def generate_code():
    """Generate Python code from neural network definition"""
    try:
        json_data = request.get_json()
        
        if not json_data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Generate Python code - no need to flatten data, generator handles it internally
        python_code = generate(json_data)
        
        return jsonify({
            "success": True,
            "python_code": python_code
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/parse', methods=['POST'])
def parse_nodes():
    """Validate neural network structure and connections"""
    try:
        json_data = request.get_json()
        if not json_data or "nodes" not in json_data:
            return jsonify({"error": "Missing 'nodes' field in request"}), 400
            
        errors_info = parse(json_data["nodes"])
        
        return jsonify({
            "info": errors_info,
        }), 200
        
    except Exception as e:
        traceback.print_exception(e)
        return jsonify({"error": str(e)}), 500
    


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/api/operations', methods=['GET'])
def get_all_operations():
    """Get all operation definitions in one call"""
    try:
        if not db:
            return jsonify({"error": "Database not initialized"}), 500
            
        return jsonify({
            "layers": db.defaults.get("layers", {"data": []}),
            "tensorOps": db.defaults.get("tensorOperations", {"data": []}),
            "activators": db.defaults.get("activationFunction", {"data": []}),
            "inputs": db.defaults.get("inputs", {"data": []})
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/operations/<operation_type>', methods=['GET'])
def get_operations_by_type(operation_type):
    """Get specific operation type definitions (layers, tensorops, activators)"""
    try:
        if not db:
            return jsonify({"error": "Database not initialized"}), 500
        
        # Map frontend names to backend keys
        type_mapping = {
            "layers": "layers",
            "tensorops": "tensorOperations", 
            "activators": "activationFunction",
            "inputs": "inputs"
        }
        
        if operation_type not in type_mapping:
            return jsonify({"error": f"Invalid operation type: {operation_type}"}), 400
            
        db_key = type_mapping[operation_type]
        if db.defaults.get(db_key):
            return jsonify(db.defaults[db_key]), 200
        else:
            return jsonify({"error": f"{operation_type} definitions not available"}), 500
            
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