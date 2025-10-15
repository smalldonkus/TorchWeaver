import json

def generate(json_data):
    """
    Generate PyTorch neural network code from JSON specification
    """
    # Extract network information
    inputs = json_data.get("inputs", ["input"])
    layers = json_data.get("layers", [])
    outputs = json_data.get("outputs", [])
    
    # Generate the complete Python file
    imports = generate_imports()
    class_definition = generate_class(layers, inputs, outputs)
    
    return imports + "\n\n" + class_definition

def generate_imports():
    """Generate the necessary import statements"""
    return """import torch
import torch.nn as nn
import torch.nn.functional as F"""

def generate_class(layers, inputs, outputs):
    """Generate the neural network class"""
    # Extract input shape information
    input_info = extract_input_info(inputs)
    
    # Generate class header
    class_name = "GeneratedNetwork"
    header = f"""class {class_name}(nn.Module):
    def __init__(self):
        super({class_name}, self).__init__()"""
    
    # Generate layer definitions
    layer_definitions = generate_layer_definitions(layers)
    
    # Generate forward pass
    forward_method = generate_forward_method(layers, inputs, outputs)
    
    # Combine all parts
    init_method = header + "\n" + layer_definitions
    complete_class = init_method + "\n" + forward_method
    
    return complete_class

def extract_input_info(inputs):
    """Extract input shape information from inputs array"""
    input_info = {}
    if len(inputs) >= 2:
        input_name = inputs[0]
        shape_str = inputs[1] if len(inputs) > 1 else "unknown"
        input_info[input_name] = shape_str
    elif len(inputs) == 1:
        input_info[inputs[0]] = "unknown"
    
    return input_info

def generate_layer_definitions(layers):
    """Generate the layer definitions in __init__ method"""
    layer_defs = []
    
    for layer in layers:
        layer_id = layer.get("id", "unknown")
        layer_type = layer.get("type", "Unknown")
        parameters = layer.get("parameters", {})
        
        # Create valid Python variable name
        var_name = sanitize_variable_name(layer_id)
        
        # Generate parameter string
        if parameters:
            param_strs = []
            for key, value in parameters.items():
                if isinstance(value, bool):
                    param_strs.append(f"{key}={str(value)}")
                elif isinstance(value, str):
                    param_strs.append(f"{key}='{value}'")
                else:
                    param_strs.append(f"{key}={value}")
            
            param_string = ", ".join(param_strs)
            layer_def = f"        self.{var_name} = nn.{layer_type}({param_string})"
        else:
            layer_def = f"        self.{var_name} = nn.{layer_type}()"
        
        layer_defs.append(layer_def)
    
    return "\n".join(layer_defs)

def generate_forward_method(layers, inputs, outputs):
    """Generate the forward method based on layer connections"""
    method_header = """
    def forward(self, x):"""
    
    # Create a mapping of layer outputs to their variable names
    output_map = {}
    
    # Add input to the output map
    if inputs:
        input_name = inputs[0] if inputs else "input"
        output_map[input_name] = "x"
    
    forward_steps = []
    
    for layer in layers:
        layer_id = layer.get("id", "unknown")
        layer_inputs = layer.get("inputs", [])
        layer_outputs = layer.get("outputs", [])
        
        # Create valid Python variable names
        var_name = sanitize_variable_name(layer_id)
        output_var = sanitize_variable_name(layer_outputs[0]) if layer_outputs else f"{var_name}_out"
        
        # Determine input variable
        if layer_inputs:
            input_var = output_map.get(layer_inputs[0], "x")
        else:
            input_var = "x"
        
        # Generate forward step
        forward_step = f"        {output_var} = self.{var_name}({input_var})"
        forward_steps.append(forward_step)
        
        # Update output map
        if layer_outputs:
            output_map[layer_outputs[0]] = output_var
    
    # Add return statement
    if outputs:
        final_output = output_map.get(outputs[0], forward_steps[-1].split(" = ")[0].strip() if forward_steps else "x")
        return_statement = f"        return {final_output}"
    else:
        # Return the last computed value
        if forward_steps:
            last_var = forward_steps[-1].split(" = ")[0].strip()
            return_statement = f"        return {last_var}"
        else:
            return_statement = "        return x"
    
    # Combine all parts
    forward_body = "\n".join(forward_steps)
    complete_forward = method_header + "\n" + forward_body + "\n" + return_statement
    
    return complete_forward

def sanitize_variable_name(name):
    """Convert layer ID to valid Python variable name"""
    # Replace invalid characters with underscores
    import re
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
    
    # Ensure it starts with a letter or underscore
    if sanitized and sanitized[0].isdigit():
        sanitized = "layer_" + sanitized
    
    return sanitized if sanitized else "unknown_layer"

# Keep the original test code for compatibility
if __name__ == "__main__":
    f = open("JsonLayers/test.json", "r")
    data = json.load(f)
    f.close()
    print(generate(data))
