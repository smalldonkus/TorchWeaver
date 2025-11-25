import unittest
from parse import parse, Graph, ParseError, find
from NNdatabase import NNDataBase

# note that these nodes are simplified versions of actual nodes used in the application
class TestParseBasics(unittest.TestCase):
    """Test suite for basic parse validation (inputs, outputs, structure)"""
    
    def setUp(self):
        """Initialize database for tests"""
        self.db = NNDataBase()
    
    def _make_node(self, node_id, op_type, node_type, parents=None, children=None, in_ch=None, out_ch=None):
        """Helper to create a properly formatted node"""
        return {
            "id": node_id,
            "data": {
                "operationType": op_type,
                "type": node_type,
                "inputChannels": in_ch,
                "outputChannels": out_ch
            },
            "parents": parents or [],
            "children": children or []
        }
    
    def test_valid_simple_network(self):
        """Test parsing a valid simple network returns no errors"""
        nodes = [
            self._make_node("input1", "Input", "SingleDimensionalInput", children=["layer1"], out_ch=10),
            self._make_node("layer1", "Layer", "Linear", parents=["input1"], children=["output1"], in_ch=10, out_ch=5),
            self._make_node("output1", "Output", "Output", parents=["layer1"])
        ]
        
        errors = parse(nodes)
        self.assertEqual(len(errors), 0, f"Expected no errors but got: {errors}")
    
    def test_missing_input_error(self):
        """Test that missing input node is detected"""
        nodes = [
            self._make_node("layer1", "Layer", "Linear", children=["output1"], in_ch=10, out_ch=5),
            self._make_node("output1", "Output", "Output", parents=["layer1"])
        ]
        
        errors = parse(nodes)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("Input" in err["errorMsg"] for err in errors))
    
    def test_missing_output_error(self):
        """Test that missing output node is detected"""
        nodes = [
            self._make_node("input1", "Input", "SingleDimensionalInput", children=["layer1"], out_ch=10),
            self._make_node("layer1", "Layer", "Linear", parents=["input1"], in_ch=10, out_ch=5)
        ]
        
        errors = parse(nodes)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("Output" in err["errorMsg"] for err in errors))
    
    def test_input_with_parent_error(self):
        """Test that input nodes with parents are rejected"""
        nodes = [
            self._make_node("layer1", "Layer", "Linear", children=["input1"], in_ch=10, out_ch=10),
            self._make_node("input1", "Input", "SingleDimensionalInput", parents=["layer1"], children=["output1"], out_ch=10),
            self._make_node("output1", "Output", "Output", parents=["input1"])
        ]
        
        errors = parse(nodes)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("parent" in err["errorMsg"].lower() for err in errors))
    
    def test_multiple_outputs_error(self):
        """Test that multiple output nodes are rejected"""
        nodes = [
            self._make_node("input1", "Input", "SingleDimensionalInput", children=["output1", "output2"], out_ch=10),
            self._make_node("output1", "Output", "Output", parents=["input1"]),
            self._make_node("output2", "Output", "Output", parents=["input1"])
        ]
        
        errors = parse(nodes)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("many outputs" in err["errorMsg"].lower() for err in errors))
    
    def test_input_with_parent_flags_correct_node(self):
        """Test that input with parent error flags the correct node ID"""
        nodes = [
            self._make_node("layer1", "Layer", "Linear", children=["input1"], in_ch=10, out_ch=10),
            self._make_node("input1", "Input", "SingleDimensionalInput", parents=["layer1"], children=["output1"], out_ch=10),
            self._make_node("output1", "Output", "Output", parents=["input1"])
        ]
        
        errors = parse(nodes)
        # Find the error about input having a parent
        parent_errors = [err for err in errors if "parent" in err["errorMsg"].lower() and "input" in err["errorMsg"].lower()]
        self.assertGreater(len(parent_errors), 0)
        # Check that input1 is flagged
        self.assertIn("input1", parent_errors[0]["flaggedNodes"])
    
    def test_missing_input_has_no_flagged_nodes(self):
        """Test that missing input error has no specific node flagged"""
        nodes = [
            self._make_node("layer1", "Layer", "Linear", children=["output1"], in_ch=10, out_ch=5),
            self._make_node("output1", "Output", "Output", parents=["layer1"])
        ]
        
        errors = parse(nodes)
        input_errors = [err for err in errors if "Input" in err["errorMsg"]]
        self.assertGreater(len(input_errors), 0)
        # Missing input is a global error, so flaggedNodes should be None
        self.assertIsNone(input_errors[0]["flaggedNodes"])
    
    def test_multiple_outputs_flags_all_output_nodes(self):
        """Test that multiple outputs error flags all output nodes"""
        nodes = [
            self._make_node("input1", "Input", "SingleDimensionalInput", children=["output1", "output2"], out_ch=10),
            self._make_node("output1", "Output", "Output", parents=["input1"]),
            self._make_node("output2", "Output", "Output", parents=["input1"])
        ]
        
        errors = parse(nodes)
        output_errors = [err for err in errors if "many outputs" in err["errorMsg"].lower()]
        self.assertGreater(len(output_errors), 0)
        # Both output nodes should be flagged
        flagged = output_errors[0]["flaggedNodes"]
        self.assertIn("output1", flagged)
        self.assertIn("output2", flagged)


class TestTarjanAlgorithm(unittest.TestCase):
    """Test suite for Tarjan's strongly connected components algorithm"""
    
    # helper used to create nodes for Tarjan tests
    def _make_node(self, node_id, children):
        """Helper to create minimal node for Tarjan tests"""
        return {
            "id": node_id,
            "children": children,
            "data": {
                "operationType": "Layer",
                "inputChannels": 10,
                "outputChannels": 10
            }
        }
    
    def test_tarjan_simple_cycle(self):
        """Test Tarjan's algorithm detects a simple cycle"""

        # Create a graph with a cycle: 1 -> 2 -> 3 -> 1
        nodes = [
            self._make_node("1", ["2"]),
            self._make_node("2", ["3"]),
            self._make_node("3", ["1"])
        ]
        
        graph = Graph(nodes)
        graph.tarjan()
        
        # There should be one SCC with all three nodes,
        # therefore indicating a cycle since all nodes can reach each other

        scc_with_cycle = [scc for scc in graph.SSCs if len(scc) > 1]
        self.assertEqual(len(scc_with_cycle), 1)
        self.assertEqual(set(scc_with_cycle[0]), {"1", "2", "3"})
    
    def test_tarjan_no_cycle(self):
        """Test Tarjan's algorithm with no cycles"""
        nodes = [
            self._make_node("1", ["2"]),
            self._make_node("2", ["3"]),
            self._make_node("3", [])
        ]
        
        graph = Graph(nodes)
        graph.tarjan()
        
        # All SCCs should be single nodes
        # since no cycles exist in the graph
        all_single_nodes = all(len(scc) == 1 for scc in graph.SSCs)
        self.assertTrue(all_single_nodes)



class TestParseError(unittest.TestCase):
    """Test suite for ParseError class"""
    
    def test_parse_error_with_nodes(self):
        """Test ParseError with node IDs"""
        error = ParseError("Test error message", nodeIDs=["node1", "node2"])
        
        self.assertEqual(error.desc, "Test error message")
        self.assertEqual(error.nodes, ["node1", "node2"])
        self.assertEqual(error.report(), "Test error message")
    
    def test_parse_error_without_nodes(self):
        """Test ParseError without node IDs"""
        error = ParseError("Global error message")
        
        self.assertEqual(error.desc, "Global error message")
        self.assertIsNone(error.nodes)
        self.assertEqual(error.report(), "Global error message")


class TestFindHelper(unittest.TestCase):
    """Test suite for find helper function"""
    
    def test_find_existing_node(self):
        """Test finding an existing node"""
        nodes = [
            {"id": "node1", "data": "value1"},
            {"id": "node2", "data": "value2"},
            {"id": "node3", "data": "value3"}
        ]
        
        result = find(nodes, "node2")
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "node2")
        self.assertEqual(result["data"], "value2")
    
    def test_find_nonexistent_node(self):
        """Test finding a non-existent node returns None"""
        nodes = [
            {"id": "node1", "data": "value1"},
            {"id": "node2", "data": "value2"}
        ]
        
        result = find(nodes, "node999")
        self.assertIsNone(result)
    
    def test_find_empty_list(self):
        """Test finding in an empty list returns None"""
        result = find([], "anynode")
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
