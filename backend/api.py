from flask import Flask, request, jsonify
from flask_cors import CORS
from NNgenerator import generate
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/convert-json-to-python', methods=['POST'])
def convert_json_to_python():
    try:
        # Get JSON data from the request
        json_data = request.get_json()
        
        if not json_data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields
        if "layers" not in json_data:
            return jsonify({"error": "Missing 'layers' field in JSON"}), 400
        
        # Use your existing generator to convert JSON to Python
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

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)