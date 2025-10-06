from flask import Flask
from routes.layer_routes import layer_bp

app = Flask(__name__)
app.register_blueprint(layer_bp)

if __name__ == "__main__":
    app.run(debug=True)