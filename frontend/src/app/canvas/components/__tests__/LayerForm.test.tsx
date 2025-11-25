import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';
import LayerForm from '../LayerForm';

// Mock the ID generator to return predictable IDs
jest.mock('@/app/canvas/utils/idGenerator', () => ({
  generateUniqueNodeId: jest.fn((prefix) => `${prefix}1`),
}));

// Track parameter updates
const mockUpdateParameters = jest.fn();
const mockHandleParameterChange = jest.fn();
const mockHandleValidationChange = jest.fn();

// Mock the parameter handling hook
jest.mock('@/app/canvas/hooks/useParameterHandling', () => ({
  useParameterHandling: () => ({
    parameters: {},
    hasValidationErrors: false,
    handleParameterChange: mockHandleParameterChange,
    handleValidationChange: mockHandleValidationChange,
    updateParameters: mockUpdateParameters,
  }),
}));

// Mock createNode but use the real implementation to verify structure
const mockCreateNode = jest.fn();
jest.mock('@/app/canvas/components/TorchNodeCreator', () => {
  const actualModule = jest.requireActual('@/app/canvas/components/TorchNodeCreator');
  return {
    ...actualModule,
    createNode: (...args: any[]) => {
      mockCreateNode(...args);
      return actualModule.createNode(...args);
    },
  };
});

describe('LayerForm', () => {
  const mockAddNode = jest.fn();
  const mockGetSetters = jest.fn(() => ({}));
  const mockGetDefaults = jest.fn(() => ({}));

  // Note we have simplified default layers for testing

  const mockDefaultLayers = {
    data: {
      'Linear Layers': {
        'Linear': {
          library: 'torch.nn',
          parameters: {
            in_features: 1,
            out_features: 1,
            bias: true
          },
          parameters_format: {
            in_features: ['int'],
            out_features: ['int'],
            bias: ['boolean']
          }
        }
      },
      'Convolutional Layers': {
        'Conv2d': {
          library: 'torch.nn',
          parameters: {
            in_channels: 1,
            out_channels: 1,
            kernel_size: 3
          },
          parameters_format: {
            in_channels: ['int'],
            out_channels: ['int'],
            kernel_size: ['int', 'tuple']
          }
        }
      }
    }
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockUpdateParameters.mockClear();
    mockCreateNode.mockClear();
    mockHandleParameterChange.mockClear();
    mockHandleValidationChange.mockClear();
  });

  describe('UI Behavior', () => {
    it('should update type dropdown when class is changed', async () => {
      render(
        <LayerForm
          nodes={[]}
          addNode={mockAddNode}
          defaultLayers={mockDefaultLayers}
          getSetters={mockGetSetters}
          getDefaults={mockGetDefaults}
        />
      );
      
      // Verify initial state - should show Linear
      await waitFor(() => {
        const typeInput = screen.getByDisplayValue('Linear');
        expect(typeInput).toBeInTheDocument();
      });
      
      // Change class to Convolutional Layers
      const classDropdown = screen.getByLabelText('Layer Class');
      await userEvent.click(classDropdown);
      const convOption = screen.getByText('Convolutional Layers');
      await userEvent.click(convOption);
      
      // Now the type dropdown should show Conv2d
      await waitFor(() => {
        const typeInput = screen.getByDisplayValue('Conv2d');
        expect(typeInput).toBeInTheDocument();
      });
    });
  });

  describe('Node Creation', () => {
    it('should create a layer node with correct structure and all parameters', async () => {
      render(
        <LayerForm
          nodes={[]}
          addNode={mockAddNode}
          defaultLayers={mockDefaultLayers}
          getSetters={mockGetSetters}
          getDefaults={mockGetDefaults}
        />
      );
      
      // Click the create button
      const createButton = screen.getByRole('button', { name: /add layer/i });
      await userEvent.click(createButton);
      
      // Verify createNode and addNode were called
      expect(mockCreateNode).toHaveBeenCalled();
      expect(mockAddNode).toHaveBeenCalled();
      
      // Verify the node has correct structure
      // operationType as 'Layer'
      // label as 'Linear'
      // and type as 'torchNode' (note this is the node type, not the data type)

      const createdNodesArray = mockAddNode.mock.calls[0][0];
      const createdNode = createdNodesArray[createdNodesArray.length - 1];
      expect(createdNode.type).toBe('torchNode');
      expect(createdNode.data.label).toBe('Linear');
      expect(createdNode.data.operationType).toBe('Layer');
    });

    it('should create two different layer types with distinct IDs and parameters', async () => {
      // Mock ID generator to return sequential IDs
      const { generateUniqueNodeId } = require('@/app/canvas/utils/idGenerator');
      generateUniqueNodeId
        .mockReturnValueOnce('layer1')
        .mockReturnValueOnce('layer2');
      
      render(
        <LayerForm
          nodes={[]}
          addNode={mockAddNode}
          defaultLayers={mockDefaultLayers}
          getSetters={mockGetSetters}
          getDefaults={mockGetDefaults}
        />
      );
      
      // Create first layer (Linear)
      const createButton = screen.getByRole('button', { name: /add layer/i });
      await userEvent.click(createButton);
      
      // Verify first node was created
      const firstCallNodesArray = mockAddNode.mock.calls[0][0];
      const firstNode = firstCallNodesArray[firstCallNodesArray.length - 1];
      expect(firstNode.id).toBe('layer1');
      expect(firstNode.data.label).toBe('Linear');
      expect(firstNode.data.operationType).toBe('Layer');
      
      // Change class to Convolutional Layers and select Conv2d
      const classDropdown = screen.getByLabelText('Layer Class');
      await userEvent.click(classDropdown);
      const convOption = screen.getByText('Convolutional Layers');
      await userEvent.click(convOption);
      
      // Create second layer (Conv2d)
      await userEvent.click(createButton);
      
      // Verify second node was created with different properties
      const secondCallNodesArray = mockAddNode.mock.calls[1][0];
      const secondNode = secondCallNodesArray[secondCallNodesArray.length - 1];
      expect(secondNode.id).toBe('layer2');
      expect(secondNode.data.label).toBe('Conv2d');
      expect(secondNode.data.operationType).toBe('Layer');
    });
  });
});
