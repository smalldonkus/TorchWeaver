import unittest
from NNgenerator import generate, generate_imports, generate_class, generate_init_method, generate_forward_method, generate_tensor_operation
from NNdatabase import NNDataBase

# Note that the nodes are heavily simplified for testing purposes

class TestNNGenerator(unittest.TestCase):
    """Test suite for neural network code generator"""
    
    def setUp(self):
        """Initialize database for tests"""
        self.db = NNDataBase()
    
    def test_generate_imports(self):
        """Test that imports are generated correctly"""
        imports = generate_imports()
        
        self.assertIn("import torch", imports)
        self.assertIn("import torch.nn as nn", imports)
        self.assertIn("import torch.nn.functional as F", imports)
    
    def test_generate_simple_network(self):
        """Test generating code for a simple input->layer->output network"""
        json_data = {
            "nodes": [
                {
                    "id": "input1",
                    "type": "SingleDimensionalInput",
                    "operation_type": "Input",
                    "parent": None,
                    "children": ["layer1"],
                    "parameters": {}
                },
                {
                    "id": "layer1",
                    "type": "Linear",
                    "operation_type": "Layer",
                    "parent": "input1",
                    "children": ["output1"],
                    "parameters": {"in_features": 10, "out_features": 5}
                },
                {
                    "id": "output1",
                    "type": "Output",
                    "operation_type": "Output",
                    "parent": "layer1",
                    "children": [],
                    "parameters": {}
                }
            ]
        }
        
        code = generate(json_data)
        
        # Check basic structure
        self.assertIn("import torch", code)
        self.assertIn("class GeneratedModel(nn.Module):", code)
        self.assertIn("def __init__(self):", code)
        self.assertIn("def forward(self, x):", code)
        
        # Check layer definition
        self.assertIn("self.layer1 = nn.Linear(in_features=10, out_features=5)", code)
        
        # Check forward pass
        self.assertIn("input1 = x", code)
        self.assertIn("layer1 = self.layer1(input1)", code)
        self.assertIn("return layer1", code)
    
    def test_generate_multiple_inputs(self):
        """Test generating code with multiple input nodes"""
        json_data = {
            "nodes": [
                {"id": "input1", "type": "SingleDimensionalInput", "operation_type": "Input", "parent": None, "children": ["merge1"], "parameters": {}},
                {"id": "input2", "type": "SingleDimensionalInput", "operation_type": "Input", "parent": None, "children": ["merge1"], "parameters": {}},
                {"id": "merge1", "type": "cat", "operation_type": "TensorOp", "parent": ["input1", "input2"], "children": [], "parameters": {"dim": 1}}
            ]
        }
        
        code = generate(json_data)
        
        # Check multiple input signature
        self.assertIn("def forward(self, x1, x2):", code)
        self.assertIn("input1 = x1", code)
        self.assertIn("input2 = x2", code)
    
    def test_generate_with_activator(self):
        """Test generating code with activation function"""
        json_data = {
            "nodes": [
                {"id": "input1", "type": "SingleDimensionalInput", "operation_type": "Input", "parent": None, "children": ["layer1"], "parameters": {}},
                {"id": "layer1", "type": "Linear", "operation_type": "Layer", "parent": "input1", "children": ["act1"], "parameters": {"in_features": 10, "out_features": 5}},
                {"id": "act1", "type": "ReLU", "operation_type": "Activator", "parent": "layer1", "children": [], "parameters": {}}
            ]
        }
        
        code = generate(json_data)
        
        # Check activation layer
        self.assertIn("self.act1 = nn.ReLU()", code)
        self.assertIn("act1 = self.act1(layer1)", code)
    
    def test_generate_tensor_merge_operation(self):
        """Test generating tensor merge operations like cat"""
        json_data = {
            "nodes": [
                {"id": "input1", "type": "SingleDimensionalInput", "operation_type": "Input", "parent": None, "children": ["cat1"], "parameters": {}},
                {"id": "input2", "type": "SingleDimensionalInput", "operation_type": "Input", "parent": None, "children": ["cat1"], "parameters": {}},
                {"id": "cat1", "type": "cat", "operation_type": "TensorOp", "parent": ["input1", "input2"], "children": [], "parameters": {"dim": 1}, "outgoing_edges_count": 0}
            ]
        }
        
        code = generate(json_data)
        
        # Check tensor operation
        self.assertIn("cat1 = torch.cat([input1, input2], dim=1)", code)
    
    def test_generate_complex_network(self):
        """Test generating a more complex network with branches"""
        json_data = {
            "nodes": [
                {"id": "input1", "type": "SingleDimensionalInput", "operation_type": "Input", "parent": None, "children": ["layer1"], "parameters": {}},
                {"id": "layer1", "type": "Linear", "operation_type": "Layer", "parent": "input1", "children": ["act1"], "parameters": {"in_features": 10, "out_features": 20}},
                {"id": "act1", "type": "ReLU", "operation_type": "Activator", "parent": "layer1", "children": ["layer2"], "parameters": {}},
                {"id": "layer2", "type": "Linear", "operation_type": "Layer", "parent": "act1", "children": ["output1"], "parameters": {"in_features": 20, "out_features": 5}},
                {"id": "output1", "type": "Output", "operation_type": "Output", "parent": "layer2", "children": [], "parameters": {}}
            ]
        }
        
        code = generate(json_data)
        
        # Verify complete flow
        self.assertIn("self.layer1 = nn.Linear(in_features=10, out_features=20)", code)
        self.assertIn("self.act1 = nn.ReLU()", code)
        self.assertIn("self.layer2 = nn.Linear(in_features=20, out_features=5)", code)
        
        self.assertIn("layer1 = self.layer1(input1)", code)
        self.assertIn("act1 = self.act1(layer1)", code)
        self.assertIn("layer2 = self.layer2(act1)", code)
        self.assertIn("return layer2", code)
    
    def test_generate_dropout_layer(self):
        """Test generating dropout layer with parameters"""
        json_data = {
            "nodes": [
                {"id": "input1", "type": "SingleDimensionalInput", "operation_type": "Input", "parent": None, "children": ["dropout1"], "parameters": {}},
                {"id": "dropout1", "type": "Dropout", "operation_type": "Layer", "parent": "input1", "children": [], "parameters": {"p": 0.5}}
            ]
        }
        
        code = generate(json_data)
        
        self.assertIn("self.dropout1 = nn.Dropout(p=0.5)", code)
        self.assertIn("dropout1 = self.dropout1(input1)", code)
    
    def test_generate_with_none_parameter(self):
        """Test handling of None parameter values"""
        json_data = {
            "nodes": [
                {"id": "input1", "type": "SingleDimensionalInput", "operation_type": "Input", "parent": None, "children": ["layer1"], "parameters": {}},
                {"id": "layer1", "type": "Linear", "operation_type": "Layer", "parent": "input1", "children": [], "parameters": {"in_features": 10, "out_features": 5, "bias": "None"}}
            ]
        }
        
        code = generate(json_data)
        
        # Check that "None" string is converted to Python None (without quotes)
        self.assertIn("bias=None", code)
        self.assertNotIn('bias="None"', code)
    
    def test_error_on_empty_nodes(self):
        """Test that error is raised when no nodes are provided"""
        json_data = {"nodes": []}
        
        with self.assertRaises(ValueError) as context:
            generate(json_data)
        
        self.assertIn("No nodes found", str(context.exception))
    
    def test_generate_conv2d_layer(self):
        """Test generating Conv2d layer"""
        json_data = {
            "nodes": [
                {"id": "input1", "type": "TwoDimensionalInput", "operation_type": "Input", "parent": None, "children": ["conv1"], "parameters": {}},
                {"id": "conv1", "type": "Conv2d", "operation_type": "Layer", "parent": "input1", "children": [], "parameters": {"in_channels": 3, "out_channels": 64, "kernel_size": 3}}
            ]
        }
        
        code = generate(json_data)
        
        self.assertIn("self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)", code)
    
    def test_generate_multiple_children(self):
        """Test generating code when a node has multiple children (branching)"""
        json_data = {
            "nodes": [
                {"id": "input1", "type": "SingleDimensionalInput", "operation_type": "Input", "parent": None, "children": ["layer1", "layer2"], "parameters": {}},
                {"id": "layer1", "type": "Linear", "operation_type": "Layer", "parent": "input1", "children": [], "parameters": {"in_features": 10, "out_features": 5}},
                {"id": "layer2", "type": "Linear", "operation_type": "Layer", "parent": "input1", "children": [], "parameters": {"in_features": 10, "out_features": 3}}
            ]
        }
        
        code = generate(json_data)
        
        # Both layers should use the same input
        self.assertIn("layer1 = self.layer1(input1)", code)
        self.assertIn("layer2 = self.layer2(input1)", code)
    

    def test_generate_without_output_node(self):
        """Test generating code when there's no explicit output node"""
        json_data = {
            "nodes": [
                {"id": "input1", "type": "SingleDimensionalInput", "operation_type": "Input", "parent": None, "children": ["layer1"], "parameters": {}},
                {"id": "layer1", "type": "Linear", "operation_type": "Layer", "parent": "input1", "children": [], "parameters": {"in_features": 10, "out_features": 5}}
            ]
        }
        
        code = generate(json_data)
        
        # Should return the last processing node
        self.assertIn("return layer1", code)
    
    def test_generate_init_method_filters_operations(self):
        """Test that init method only includes Layer and Activator nodes"""
        nodes = [
            {"id": "input1", "type": "SingleDimensionalInput", "operation_type": "Input", "parameters": {}},
            {"id": "layer1", "type": "Linear", "operation_type": "Layer", "parameters": {"in_features": 10, "out_features": 5}},
            {"id": "tensorop1", "type": "cat", "operation_type": "TensorOp", "parameters": {}},
            {"id": "act1", "type": "ReLU", "operation_type": "Activator", "parameters": {}}
        ]
        
        init_lines = generate_init_method(nodes, self.db)
        init_code = "\n".join(init_lines)
        
        # Should include layer and activator
        self.assertIn("self.layer1", init_code)
        self.assertIn("self.act1", init_code)
        
        # Should NOT include input or tensor op
        self.assertNotIn("self.input1", init_code)
        self.assertNotIn("self.tensorop1", init_code)


if __name__ == '__main__':
    unittest.main()
