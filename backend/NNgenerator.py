import json
from collections import defaultdict

def generate(json_data, tensor_ops_config=None):
    """
    Generate PyTorch neural network code from JSON specification
    
    Args:
        json_data: The network specification JSON
        tensor_ops_config: Optional tensor operations configuration from TensorOps.json
    """
    # Extract network information
    inputs = json_data.get("inputs", ["input"])
    nodes = json_data.get("nodes", json_data.get("layers", []))  # Support both formats
    outputs = json_data.get("outputs", [])
    
    # Load tensor ops config if not provided
    if tensor_ops_config is None:
        tensor_ops_config = load_tensor_ops_config()
    
    # Generate the complete Python file
    imports = generate_imports()
    class_definition = generate_class(nodes, inputs, outputs, tensor_ops_config)
    
    return imports + "\n\n" + class_definition

def load_tensor_ops_config():
    """Load tensor operations configuration from TensorOps.json"""
    try:
        with open("JsonLayers/TensorOps.json", "r") as f:
            data = json.load(f)
            return data.get("data", [])
    except FileNotFoundError:
        print("Warning: TensorOps.json not found, using empty config")
        return []

def generate_imports():
    """Generate the necessary import statements"""
    return """import torch
import torch.nn as nn
import torch.nn.functional as F"""

def generate_class(nodes, inputs, outputs, tensor_ops_config):
    """Generate the neural network class with multi-branch support"""
    # Extract input shape information
    input_info = extract_input_info(inputs)
    
    # Organize nodes by branch
    branches = organize_nodes_by_branch(nodes)
    
    # Generate class header
    class_name = "MultiBranchNet"
    header = f"""class {class_name}(nn.Module):
    def __init__(self):
        super({class_name}, self).__init__()"""
    
    # Generate layer definitions organized by branch
    layer_definitions = generate_multi_branch_layer_definitions(branches)
    
    # Generate forward pass
    forward_method = generate_multi_branch_forward_method(branches, inputs, outputs, tensor_ops_config)
    
    # Combine all parts
    init_method = header + "\n" + layer_definitions
    complete_class = init_method + "\n" + forward_method
    
    return complete_class

def organize_nodes_by_branch(nodes):
    """Organize nodes by their branch field"""
    branches = defaultdict(list)
    tensor_ops = []
    
    for node in nodes:
        operation_type = node.get("operation_type", "layer")
        branch = node.get("branch", "branch_1")
        
        if operation_type == "tensor_op":
            tensor_ops.append(node)
        else:
            branches[branch].append(node)
    
    # Add tensor operations to a special category
    if tensor_ops:
        branches["tensor_ops"] = tensor_ops
    
    return dict(branches)

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

def generate_multi_branch_layer_definitions(branches):
    """Generate layer definitions organized by branch"""
    layer_defs = []
    
    # Process branches in order
    sorted_branches = sorted(branches.keys())
    
    for branch_name in sorted_branches:
        nodes = branches[branch_name]
        
        if branch_name == "tensor_ops":
            # Skip tensor operations in __init__ - they don't need layer definitions
            continue
        
        # Add branch comment
        if len(branches) > 1 and branch_name != "tensor_ops":
            layer_defs.append(f"\n        # --- {branch_name.replace('_', ' ').title()} layers ---")
        
        # Generate layer definitions for this branch
        for node in nodes:
            layer_id = node.get("id", "unknown")
            layer_type = node.get("type", "Unknown")
            parameters = node.get("parameters", {})
            
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

def generate_multi_branch_forward_method(branches, inputs, outputs, tensor_ops_config):
    """Generate the forward method with multi-branch support"""
    method_header = """
    def forward(self, x):"""
    
    # Create a mapping of node outputs to their variable names
    output_map = {}
    
    # Add input to the output map
    if inputs:
        input_name = inputs[0] if inputs else "input"
        output_map[input_name] = "x"
    
    # Flatten all nodes from all branches
    all_nodes = []
    for branch_name, nodes in branches.items():
        for node in nodes:
            node_with_branch = node.copy()
            node_with_branch["_branch_name"] = branch_name
            all_nodes.append(node_with_branch)
    
    # Build dependency graph
    node_by_output = {}
    for node in all_nodes:
        for output in node.get("outputs", []):
            node_by_output[output] = node
    
    # Topological sort (simplified for DAG)
    visited = set()
    sorted_nodes = []
    
    def visit(node):
        node_id = node.get("id")
        if node_id in visited:
            return
        
        visited.add(node_id)
        
        # Visit dependencies first
        for input_ref in node.get("inputs", []):
            if input_ref in node_by_output:
                dep_node = node_by_output[input_ref]
                visit(dep_node)
        
        sorted_nodes.append(node)
    
    # Start topological sort from nodes with no unvisited dependencies
    for node in all_nodes:
        if node.get("id") not in visited:
            visit(node)
    
    forward_steps = []
    current_branch = None
    tensor_ops_added = False
    
    # Process nodes in dependency order
    for node in sorted_nodes:
        branch_name = node.get("_branch_name", "unknown")
        operation_type = node.get("operation_type", "layer")
        
        # Handle tensor operations
        if operation_type == "tensor_op":
            if not tensor_ops_added:
                forward_steps.append(f"\n        # ----- Tensor Operations -----")
                tensor_ops_added = True
            
            tensor_steps = generate_tensor_operations([node], output_map, tensor_ops_config)
            forward_steps.extend([step for step in tensor_steps if not step.startswith("\n        # -----")])
            continue
        
        # Add branch comment when switching branches (for regular layers only)
        if branch_name != current_branch:
            if branch_name != "tensor_ops":
                formatted_branch = branch_name.replace("_", " ").title()
                forward_steps.append(f"\n        # ----- {formatted_branch} -----")
            current_branch = branch_name
        
        # Handle regular layers
        layer_id = node.get("id", "unknown")
        layer_inputs = node.get("inputs", [])
        layer_outputs = node.get("outputs", [])
        
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

def build_tensor_operation_registry(tensor_ops_config):
    """
    Build operation registry from TensorOps.json configuration
    
    Args:
        tensor_ops_config: List of tensor operation definitions from TensorOps.json
        
    Returns:
        Dict mapping operation type to configuration
    """
    registry = {}
    
    for op_def in tensor_ops_config:
        op_type = op_def.get("type", "").lower()
        
        if op_type:
            # Store the complete operation definition
            registry[op_type] = op_def
    
    return registry

def generate_tensor_operations(tensor_ops, output_map, tensor_ops_config):
    """
    Generate code for tensor operations using configuration from TensorOps.json
    
    Parameter Sources:
    1. "parameters" in JSON node: Actual user values (e.g., {"dim": 1})
    
    We use user values from "parameters" for any parameter listed in "params"
    """
    forward_steps = []
    
    if not tensor_ops:
        return forward_steps
    
    # Add comment for tensor operations section
    forward_steps.append(f"\n        # ----- Tensor Operations -----")
    
    # Build the operation registry from the configuration
    op_registry = build_tensor_operation_registry(tensor_ops_config)
    
    for op in tensor_ops:
        op_type = op.get("type", "unknown")
        op_id = op.get("id", "unknown")
        op_inputs = op.get("inputs", [])
        op_outputs = op.get("outputs", [])
        op_params = op.get("parameters", {})  # User's actual parameter values
        
        # Create output variable name
        output_var = sanitize_variable_name(op_outputs[0]) if op_outputs else f"{op_id}_out"
        
        op_config = op_registry[op_type.lower()]  # Configuration from TensorOps.json
        
        # Prepare input variables
        input_vars = []
        for inp in op_inputs:
            var_name = output_map.get(inp, inp)
            input_vars.append(var_name)
        
        # Build the function call based on operation configuration
        function_call = generate_tensor_function_call(
            op_config, input_vars, op_params, output_var
        )
        
        if function_call:
            forward_steps.append(function_call)
            
            # Update output map
            if op_outputs:
                output_map[op_outputs[0]] = output_var
    
    return forward_steps

def generate_tensor_function_call(op_config, input_vars, op_params, output_var):
    """Generate a tensor operation function call based on configuration"""
    library = op_config.get('library', 'torch')
    op_type = op_config.get('type')
    code_gen = op_config.get('codeGeneration', {})
    input_format = code_gen.get('inputFormat', 'separate')
    default_params = op_config.get('parameters', {})
    
    # Merge default parameters with user parameters
    all_params = {**default_params}
    if op_params:
        all_params.update(op_params)
    
    # Build parameter list
    param_parts = []
    for param_name, param_value in all_params.items():
        if param_name != 'operation_type':
            if isinstance(param_value, str):
                param_parts.append(f"{param_name}='{param_value}'")
            else:
                param_parts.append(f"{param_name}={param_value}")
    
    # Handle input formatting
    if input_format == "tuple":
        if len(input_vars) > 1:
            inputs_str = ", ".join(input_vars)
            formatted_inputs = f"({inputs_str})"
        elif len(input_vars) == 1:
            formatted_inputs = input_vars[0]
        else:
            formatted_inputs = "()"
    else:  # separate
        formatted_inputs = ", ".join(input_vars)
    
    # Build complete function call
    if param_parts:
        if formatted_inputs:
            call_params = f"{formatted_inputs}, {', '.join(param_parts)}"
        else:
            call_params = ", ".join(param_parts)
    else:
        call_params = formatted_inputs
    
    return f"        {output_var} = {library}.{op_type}({call_params})"

def sanitize_variable_name(name):
    """Convert layer ID to valid Python variable name"""
    # Replace invalid characters with underscores
    import re
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
    
    # Ensure it starts with a letter or underscore
    if sanitized and sanitized[0].isdigit():
        sanitized = "layer_" + sanitized
    
    return sanitized if sanitized else "unknown_layer"


