import json
import os
from NNdatabase import NNDataBase

def generate(json_data, tensor_ops_config=None):
    """
    Generate PyTorch neural network code from topologically sorted JSON specification
    
    Args:
        json_data: The network specification JSON with 'nodes' array
        tensor_ops_config: Optional (not used, kept for compatibility)
    """
    # Load operation definitions from database
    db = NNDataBase()
    
    # Extract nodes (already topologically sorted)
    nodes = json_data.get("nodes", [])
    
    if not nodes:
        raise ValueError("No nodes found in JSON data")
    
    # Count input nodes for forward method signature
    input_nodes = [node for node in nodes if node.get("operation_type") == "Input"]
    
    # Generate the complete Python code
    imports = generate_imports()
    class_code = generate_class(nodes, input_nodes, db)
    
    return imports + "\n\n" + class_code

def generate_imports():
    """Generate the necessary import statements"""
    return """import torch
import torch.nn as nn
import torch.nn.functional as F"""

def generate_class(nodes, input_nodes, db):
    """Generate the neural network class"""
    class_lines = []
    class_lines.append("class GeneratedModel(nn.Module):")
    class_lines.append("    def __init__(self):")
    class_lines.append("        super(GeneratedModel, self).__init__()")
    class_lines.append("")
    
    # Generate layer definitions in __init__ (only for Layer/Activator nodes)
    init_lines = generate_init_method(nodes, db)
    class_lines.extend(init_lines)
    
    class_lines.append("")
    
    # Generate forward method signature based on number of inputs
    if len(input_nodes) == 1:
        class_lines.append("    def forward(self, x):")
    else:
        input_params = ", ".join([f"x{i+1}" for i in range(len(input_nodes))])
        class_lines.append(f"    def forward(self, {input_params}):")
    
    # Generate forward pass
    forward_lines = generate_forward_method(nodes, input_nodes, db)
    class_lines.extend(forward_lines)
    
    # Add return statement for final output
    output_nodes = [node for node in nodes if node.get("operation_type") == "Output"]
    if output_nodes:
        # Find the parent of the output node
        output_parent = output_nodes[0].get("parent")
        if output_parent:
            class_lines.append(f"        return {output_parent}")
    else:
        # Find the last non-output node
        processing_nodes = [node for node in nodes if node.get("operation_type") != "Output"]
        if processing_nodes:
            last_node = processing_nodes[-1]
            class_lines.append(f"        return {last_node['id']}")
    
    return "\n".join(class_lines)

def generate_init_method(nodes, db):
    """Generate the __init__ method layer definitions"""
    lines = []
    
    for node in nodes:
        operation_type = node.get("operation_type")
        
        # Only generate layer definitions for Layer and Activator nodes
        if operation_type in ["Layer", "Activator"]:
            node_type = node.get("type")
            node_id = node.get("id")
            parameters = node.get("parameters", {})
            
            # Find definition in database
            layer_def = find_layer_definition(node_type, db)
            
            if layer_def:
                # Generate parameter string
                param_strings = []
                for param_name, param_value in parameters.items():
                    param_strings.append(f"{param_name}={param_value}")
                
                param_str = ", ".join(param_strings)
                
                # Generate layer definition
                if layer_def.get("library") == "torch.nn":
                    lines.append(f"        self.{node_id} = nn.{node_type}({param_str})")
                else:
                    lines.append(f"        self.{node_id} = {node_type}({param_str})")
    
    return lines

def generate_forward_method(nodes, input_nodes, db):
    """Generate the forward method"""
    lines = []
    
    for node in nodes:
        node_id = node.get("id")
        node_type = node.get("type")
        operation_type = node.get("operation_type")
        parent = node.get("parent")
        
        if operation_type == "Input":
            # Input nodes assign the corresponding input parameter
            if len(input_nodes) == 1:
                lines.append(f"        {node_id} = x")
            else:
                # Find which input index this node corresponds to
                input_index = input_nodes.index(node)
                lines.append(f"        {node_id} = x{input_index + 1}")
        
        elif operation_type in ["Layer", "Activator"]:
            # Layer/Activator nodes call the defined layer
            if parent:
                if isinstance(parent, list):
                    # Multiple inputs (shouldn't happen for layers, but handle it)
                    parent_str = parent[0]  # Use first parent
                else:
                    parent_str = parent
                lines.append(f"        {node_id} = self.{node_id}({parent_str})")
        
        elif operation_type == "TensorOp":
            # Generate tensor operations using configuration
            tensor_code = generate_tensor_operation(node, db)
            if tensor_code:
                lines.append(tensor_code)
        
        elif operation_type == "Output":
            # Output nodes don't generate any code, just mark the final output
            pass
    
    return lines

def generate_tensor_operation(node, db):
    """Generate code for tensor operations using hierarchical database lookup"""
    node_id = node.get("id")
    node_type = node.get("type")
    parent = node.get("parent")
    user_parameters = node.get("parameters", {})
    outgoing_edges_count = node.get("outgoing_edges_count", 0)
    
    # Find the tensor operation configuration using hierarchical lookup
    op_config = db.find_in_category(node_type, "tensorOperations")
    if not op_config:
        # Fallback to searching all categories
        op_config = db.find_definition(node_type)
    
    if not op_config:
        # Default behavior if not found
        library = "torch"
        parent_parameter_format = "separate"
        operation_pattern = "merge"
        default_parameters = {}
    else:
        library = op_config.get("library", "torch")
        code_gen = op_config.get("codeGeneration", {})
        parent_parameter_format = code_gen.get("parentParameterFormat", "separate")
        operation_pattern = code_gen.get("operationPattern", "merge")
        default_parameters = op_config.get("parameters", {})
    
    # Merge default parameters with user parameters
    all_parameters = {**default_parameters, **user_parameters}
    
    # Handle parent inputs
    if not parent:
        return None
    
    if isinstance(parent, list):
        input_vars = parent
    else:
        input_vars = [parent]
    
    # Format inputs based on configuration
    if parent_parameter_format == "tuple":
        # For operations like torch.cat that expect a list/tuple of tensors
        formatted_inputs = f"[{', '.join(input_vars)}]"
    else:  # separate
        # For operations that take separate arguments
        formatted_inputs = ', '.join(input_vars)
    
    # Generate parameter string
    param_strings = []
    for param_name, param_value in all_parameters.items():
        if isinstance(param_value, str):
            param_strings.append(f"{param_name}='{param_value}'")
        else:
            param_strings.append(f"{param_name}={param_value}")
    
    # Build the complete function call
    if param_strings:
        if formatted_inputs:
            call_params = f"{formatted_inputs}, {', '.join(param_strings)}"
        else:
            call_params = ', '.join(param_strings)
    else:
        call_params = formatted_inputs
    
    # Handle split operations specially
    if operation_pattern == "split":
        # Generate multiple assignments for split outputs
        suffixes = ['a', 'b', 'c', 'd', 'e', 'f']  # Support up to 6 outputs
        if outgoing_edges_count > 0:
            output_vars = [f"{node_id}{suffixes[i]}" for i in range(outgoing_edges_count)]
            output_list = ", ".join(output_vars)
            return f"        {output_list} = {library}.{node_type}({call_params})"
        else:
            # Fallback if outgoing_edges_count is not available
            return f"        {node_id}_outputs = {library}.{node_type}({call_params})"
    else:
        # Regular tensor operation
        return f"        {node_id} = {library}.{node_type}({call_params})"

def find_layer_definition(layer_type, db):
    """Find layer definition in database using hierarchical lookup"""
    return db.find_definition(layer_type)


