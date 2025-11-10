import json

from enum import Enum

class NNDataBase():

    class NNLayerTypes(Enum):
        layers = "layers"
        tensorOperations = "tensorOperations"
        activationFunction = "activationFunction"
        inputs = "inputs"

    def __init__(self):
        
        self.defaults = {
            "layers" : None,
            "tensorOperations" : None,
            "activationFunction" : None,
            "inputs" : None
        }

        with open("JsonLayers/Layers.json") as f:
            self.defaults["layers"] = json.load(f)
        with open("JsonLayers/TensorOperations.json") as f:
            self.defaults["tensorOperations"] = json.load(f)
        with open("JsonLayers/ActivationFunction.json") as f:
            self.defaults["activationFunction"] = json.load(f)
        with open("JsonLayers/Input.json") as f:
            self.defaults["inputs"] = json.load(f)
        
    def find_definition(self, target_type):
        """Find a definition across all operation types using hierarchical lookup"""
        # Search in layers
        if self.defaults.get("layers") and "data" in self.defaults["layers"]:
            for class_name, class_items in self.defaults["layers"]["data"].items():
                if target_type in class_items:
                    return {
                        "type": target_type,
                        "class": class_name,
                        "category": "layers",
                        **class_items[target_type]
                    }
        
        # Search in tensor operations
        if self.defaults.get("tensorOperations") and "data" in self.defaults["tensorOperations"]:
            for class_name, class_items in self.defaults["tensorOperations"]["data"].items():
                if target_type in class_items:
                    return {
                        "type": target_type,
                        "class": class_name,
                        "category": "tensorOperations",
                        **class_items[target_type]
                    }
        
        # Search in activation functions
        if self.defaults.get("activationFunction") and "data" in self.defaults["activationFunction"]:
            for class_name, class_items in self.defaults["activationFunction"]["data"].items():
                if target_type in class_items:
                    return {
                        "type": target_type,
                        "class": class_name,
                        "category": "activationFunction",
                        **class_items[target_type]
                    }
        
        # Search in inputs
        if self.defaults.get("inputs") and "data" in self.defaults["inputs"]:
            for class_name, class_items in self.defaults["inputs"]["data"].items():
                if target_type in class_items:
                    return {
                        "type": target_type,
                        "class": class_name,
                        "category": "inputs",
                        **class_items[target_type]
                    }
        
        return None
    
    def find_in_category(self, target_type, category):
        """Find a definition in a specific category (layers, tensorOperations, activationFunction)"""
        if self.defaults.get(category) and "data" in self.defaults[category]:
            for class_name, class_items in self.defaults[category]["data"].items():
                if target_type in class_items:
                    return {
                        "type": target_type,
                        "class": class_name,
                        "category": category,
                        **class_items[target_type]
                    }
        return None
        

if __name__ == "__main__":
    pass