from flask import Blueprint, request, jsonify
from controllers.layer_controller import (
    create_network,
    add_layer,
    remove_layer,
    modify_layer,
    add_link,
    remove_link,
    get_layer_defaults,
)

layer_bp = Blueprint('layer_bp', __name__)

@layer_bp.route("/")
def home():
    return "TorchWeaver backend is running!"

@layer_bp.route("/api/networks", methods=["POST"])
def api_create_network():
    return jsonify(create_network())

@layer_bp.route("/api/networks/<network_id>/layers", methods=["POST"])
def api_add_layer(network_id):
    
    data = request.json

    return jsonify(add_layer(
        network_id,
        data.get("layer_type"),
        data.get("layer_name"),
        data.get("parameters"),
        data.get("in_layers"),
        data.get("out_layers"),
    ))

@layer_bp.route("/api/networks/<network_id>/layers/<layer_id>", methods=["DELETE"])
def api_remove_layer(network_id, layer_id):

    return jsonify(remove_layer(network_id, layer_id))

@layer_bp.route("/api/networks/<network_id>/layers/<layer_id>", methods=["PATCH"])
def api_modify_layer(network_id, layer_id):

    data = request.json

    return jsonify(modify_layer(
        network_id,
        layer_id,
        data.get("key"),
        data.get("value"),
    ))

@layer_bp.route("/api/networks/<network_id>/links", methods=["POST"])
def api_add_link(network_id):

    data = request.json

    return jsonify(add_link(
        network_id,
        data.get("a_layer_id"),
        data.get("b_layer_id"),
    ))

@layer_bp.route("/api/networks/<network_id>/links", methods=["DELETE"])
def api_remove_link(network_id):

    data = request.json

    return jsonify(remove_link(
        network_id,
        data.get("a_layer_id"),
        data.get("b_layer_id"),
    ))

@layer_bp.route("/api/layer_defaults", methods=["GET"])
def api_get_layer_defaults():

    layer_type = request.args.get("layer_type")

    layer_name = request.args.get("layer_name")
    
    return jsonify(get_layer_defaults(layer_type, layer_name))