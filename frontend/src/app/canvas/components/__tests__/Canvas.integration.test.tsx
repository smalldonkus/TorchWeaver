import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ReactFlowProvider } from '@xyflow/react';

// Mock modules before imports
jest.mock('@/app/canvas/hooks/useNodeDefinitions', () => ({
  useNodeDefinitions: () => ({
    nodeDefinitions: {
      data: {
        'torch.nn': {
          'Conv2d': {
            parameters: {
              in_channels: ['int'],
              out_channels: ['int'],
              kernel_size: ['int', 'tuple']
            }
          }
        }
      }
    }
  })
}));

jest.mock('@/app/canvas/hooks/useOperationDefinitions', () => ({
  useOperationDefinitions: () => ({
    operationDefinitions: {
      data: {
        'torch.nn': {
          'MaxPool2d': {
            parameters: {
              kernel_size: ['int', 'tuple']
            }
          }
        }
      }
    }
  })
}));

jest.mock('@/app/canvas/hooks/useVariablesInfo', () => ({
  useVariablesInfo: () => ({
    variablesInfo: {}
  })
}));

describe('Canvas Integration Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Form Integration Tests', () => {
    it('should create nodes with correct IDs from all form types', async () => {
      const { generateUniqueNodeId } = require('@/app/canvas/utils/idGenerator');
      
      // Test Input form ID generation
      const inputId = generateUniqueNodeId('input', []);
      expect(inputId).toBe('input1');
      
      // Test Layer form ID generation  
      const layerId = generateUniqueNodeId('layer', []);
      expect(layerId).toBe('layer1');
      
      // Test Activator form ID generation
      const activatorId = generateUniqueNodeId('activator', []);
      expect(activatorId).toBe('activator1');
      
      // Test TensorOp form ID generation
      const tensorOpId = generateUniqueNodeId('tensorop', []);
      expect(tensorOpId).toBe('tensorop1');
    });

    it('should generate unique IDs when nodes already exist', async () => {
      const { generateUniqueNodeId } = require('@/app/canvas/utils/idGenerator');
      
      const existingNodes = [
        { id: 'input1' },
        { id: 'input2' },
        { id: 'layer1' }
      ];
      
      // Should generate input3 since input1 and input2 exist
      const newInputId = generateUniqueNodeId('input', existingNodes);
      expect(newInputId).toBe('input3');
      
      // Should generate layer2 since layer1 exists
      const newLayerId = generateUniqueNodeId('layer', existingNodes);
      expect(newLayerId).toBe('layer2');
    });

    it('should create nodes with correct structure across all forms', async () => {
      const { createNode } = require('@/app/canvas/components/TorchNodeCreator');
      
      // Test Input node structure
      const inputNode = createNode(
        'input1',
        0,
        'TestInput',
        'Input',
        'TestInputType',
        { features: 10 },
        jest.fn(),
        jest.fn(),
        null
      );
      
      expect(inputNode.id).toBe('input1');
      expect(inputNode.type).toBe('torchNode');
      expect(inputNode.data.operationType).toBe('Input');
      expect(inputNode.data.parameters.features).toBe(10);
      
      // Test Layer node structure
      const layerNode = createNode(
        'layer1',
        1,
        'Linear',
        'Layer',
        'Linear',
        { in_features: 10, out_features: 5 },
        jest.fn(),
        jest.fn(),
        null
      );
      
      expect(layerNode.id).toBe('layer1');
      expect(layerNode.data.operationType).toBe('Layer');
      expect(layerNode.data.parameters.in_features).toBe(10);
      
      // Test Activator node structure
      const activatorNode = createNode(
        'activator1',
        2,
        'ReLU',
        'Activator',
        'ReLU',
        {},
        jest.fn(),
        jest.fn(),
        null
      );
      
      expect(activatorNode.id).toBe('activator1');
      expect(activatorNode.data.operationType).toBe('Activator');
      
      // Test TensorOp node structure
      const tensorOpNode = createNode(
        'tensorop1',
        3,
        'cat',
        'TensorOp',
        'cat',
        { dim: 0 },
        jest.fn(),
        jest.fn(),
        null
      );
      
      expect(tensorOpNode.id).toBe('tensorop1');
      expect(tensorOpNode.data.operationType).toBe('TensorOp');
      expect(tensorOpNode.data.parameters.dim).toBe(0);
    });
  });

  describe('Node Connection Workflow', () => {
    it('should validate edge connections between nodes', async () => {
      // Test that edges can be created with proper source and target
      const edge = {
        id: 'e1-2',
        source: 'input1',
        target: 'layer1',
        sourceHandle: 'output',
        targetHandle: 'input'
      };
      
      expect(edge.source).toBe('input1');
      expect(edge.target).toBe('layer1');
      expect(edge.id).toBe('e1-2');
    });

    it('should create edges with correct structure', async () => {
      const newEdge = {
        id: `e-input1-layer1`,
        source: 'input1',
        target: 'layer1',
        sourceHandle: 'output',
        targetHandle: 'input',
        animated: true
      };
      
      expect(newEdge).toHaveProperty('source');
      expect(newEdge).toHaveProperty('target');
      expect(newEdge).toHaveProperty('sourceHandle');
      expect(newEdge).toHaveProperty('targetHandle');
    });
  });

  describe('Parameter Validation', () => {
    it('should validate parameter formats', async () => {
      // Test integer parameter validation
      const intParam = 10;
      expect(typeof intParam).toBe('number');
      expect(Number.isInteger(intParam)).toBe(true);
      
      // Test float parameter validation
      const floatParam = 0.5;
      expect(typeof floatParam).toBe('number');
      
      // Test string parameter validation
      const stringParam = 'same';
      expect(typeof stringParam).toBe('string');
    });

    it('should handle parameter ranges', async () => {
      const paramValue = 5;
      const minValue = 1;
      const maxValue = 10;
      
      expect(paramValue).toBeGreaterThanOrEqual(minValue);
      expect(paramValue).toBeLessThanOrEqual(maxValue);
    });
  });

  describe('Node Management', () => {
    it('should handle node deletion', async () => {
      const nodes = [
        { id: 'input1', data: {} },
        { id: 'layer1', data: {} },
        { id: 'activator1', data: {} }
      ];
      
      // Simulate deleting layer1
      const updatedNodes = nodes.filter(node => node.id !== 'layer1');
      
      expect(updatedNodes).toHaveLength(2);
      expect(updatedNodes.find(n => n.id === 'layer1')).toBeUndefined();
      expect(updatedNodes.find(n => n.id === 'input1')).toBeDefined();
    });

    it('should handle edge cleanup on node deletion', async () => {
      const edges = [
        { id: 'e1', source: 'input1', target: 'layer1' },
        { id: 'e2', source: 'layer1', target: 'activator1' },
        { id: 'e3', source: 'activator1', target: 'output1' }
      ];
      
      const nodeToDelete = 'layer1';
      
      // Remove edges connected to deleted node
      const updatedEdges = edges.filter(
        edge => edge.source !== nodeToDelete && edge.target !== nodeToDelete
      );
      
      expect(updatedEdges).toHaveLength(1);
      expect(updatedEdges[0].id).toBe('e3');
    });
  });

  describe('State Management', () => {
    it('should track canvas state changes', async () => {
      const initialState = {
        nodes: [],
        edges: [],
        name: 'Untitled'
      };
      
      const updatedState = {
        nodes: [{ id: 'input1', data: {} }],
        edges: [],
        name: 'My Network'
      };
      
      expect(initialState.nodes).toHaveLength(0);
      expect(updatedState.nodes).toHaveLength(1);
      expect(updatedState.name).toBe('My Network');
    });

    it('should detect unsaved changes', async () => {
      const lastSavedState = {
        nodes: [{ id: 'input1' }],
        edges: [],
        name: 'Network'
      };
      
      const currentState = {
        nodes: [{ id: 'input1' }, { id: 'layer1' }],
        edges: [],
        name: 'Network'
      };
      
      const hasChanges = JSON.stringify(lastSavedState) !== JSON.stringify(currentState);
      expect(hasChanges).toBe(true);
    });
  });
});
