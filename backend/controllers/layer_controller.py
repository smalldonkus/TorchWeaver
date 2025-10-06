from models.NNmain import NN, addLayer, removeLayer, modifyLayer, addLink, removeLink, getDefaultLayerParamters

# In-memory storage for all networks will change to persistent storage later
networks = {}  # key: network_id, value: NN instance

def create_network():
    """
    Creates a new neural network instance and stores it in the global networks dictionary.

    Returns:
        dict: Contains the new network's unique ID, e.g. {"network_id": "1"}
    """
    net_id = str(len(networks) + 1)  # needs to be changed; this is too simple
    networks[net_id] = NN()
    return {"network_id": net_id}

def add_layer(network_id, layer_type, layer_name, parameters, in_layers=None, out_layers=None):
    """
    Adds a new layer to the neural network with the given network_id.

    Args:
        network_id (str): The ID of the neural network to modify.
        layer_type (str): The type of the layer
        layer_name (str): The unique name for this layer. (note might not be neccesary)
        parameters (dict): Dictionary of parameters for the layer.
        in_layers (list, optional): List of input layer IDs. (note might not be neccesary)
        out_layers (list, optional): List of output layer IDs. (note might not be neccesary)

    Returns:
        dict: Status message or error.
    """
    net = networks.get(network_id)
    if not net:
        return {"error": "Network not found"}, 404

    # You would call addLayer here, passing the network's Nodes or structure as needed
    # Example:

    # node = addLayer(layer_type, layer_name, parameters, in_layers, out_layers)
    # net.Nodes[node.ID] = node

    return {"status": "layer added"}

def remove_layer(network_id, layer_id):
    """
    Removes a layer from the neural network with the given network_id.

    Args:
        network_id (str): The ID of the neural network to modify.
        layer_id (str): The ID or name of the layer to remove.

    Returns:
        dict: Status message or error.
    """
    net = networks.get(network_id)
    if not net:
        return {"error": "Network not found"}, 404
    
    # removeLayer(layer_id) logic 
    
    return {"status": "layer removed"}

def modify_layer(network_id, layer_id, key, value):
    """
    Modifies a parameter of a specific layer in the neural network.

    Args:
        network_id (str): The ID of the neural network to modify.
        layer_id (str): The ID or name of the layer to modify.
        key (str): The parameter key to modify.
        value: The new value for the parameter.

    Returns:
        dict: Status message or error.
    """
    net = networks.get(network_id)

    if not net:
        return {"error": "Network not found"}, 404
    
    # modifyLayer(layer_id, key, value) logic here

    return {"status": "layer modified"}

def add_link(network_id, a_layer_id, b_layer_id):
    """
    Adds a connection (link) between two layers in the neural network.

    Args:
        network_id (str): The ID of the neural network to modify.
        a_layer_id (str): The source layer ID.
        b_layer_id (str): The destination layer ID.

    Returns:
        dict: Status message or error.
    """
    net = networks.get(network_id)
    if not net:
        return {"error": "Network not found"}, 404
    # addLink(a_layer_id, b_layer_id) logic here
    return {"status": "link added"}

def remove_link(network_id, a_layer_id, b_layer_id):
    """
    Removes a connection (link) between two layers in the neural network.

    Args:
        network_id (str): The ID of the neural network to modify.
        a_layer_id (str): The source layer ID.
        b_layer_id (str): The destination layer ID.

    Returns:
        dict: Status message or error.
    """
    net = networks.get(network_id)
    if not net:
        return {"error": "Network not found"}, 404
    
    # removeLink(a_layer_id, b_layer_id) logic here

    return {"status": "link removed"}

def get_layer_defaults(layer_type, layer_name):
    """
    Retrieves the default parameters for a given layer type and name.

    Args:
        layer_type (str): The type of the layer (e.g., 'conv2d', 'linear').
        layer_name (str): The name of the layer.

    Returns:
        dict: Default parameters for the layer.
    """
    return getDefaultLayerParamters(layer_type, layer_name)