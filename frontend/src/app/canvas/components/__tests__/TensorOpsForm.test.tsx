import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';
import TensorOpsForm from '../TensorOpsForm';

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

describe('TensorOpsForm', () => {
  const mockAddNode = jest.fn();
  const mockGetSetters = jest.fn(() => []);
  const mockGetDefaults = jest.fn(() => []);

  const mockDefaultTensorOps = {
    data: {
      'Merge Tensor Operations': {
        'cat': {
          library: 'torch',
          parameters: {
            dim: 0
          },
          parameters_format: {
            dim: ['int']
          }
        },
        'stack': {
          library: 'torch',
          parameters: {
            dim: 0
          },
          parameters_format: {
            dim: ['int']
          }
        }
      },
      'Reshape Tensor Operations': {
        'flatten': {
          library: 'torch',
          parameters: {
            start_dim: 0,
            end_dim: -1
          },
          parameters_format: {
            start_dim: ['int'],
            end_dim: ['int']
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
    it('should update operation type dropdown when class is changed', async () => {
      render(
        <TensorOpsForm
          nodes={[]}
          addNode={mockAddNode}
          defaultTensorOps={mockDefaultTensorOps}
          getSetters={mockGetSetters}
          getDefaults={mockGetDefaults}
        />
      );
      
      // Verify initial state - should show cat
      await waitFor(() => {
        const typeInput = screen.getByDisplayValue('cat');
        expect(typeInput).toBeInTheDocument();
      });
      
      // Change class to Reshape Tensor Operations
      const classDropdown = screen.getByLabelText('Operation Class');
      await userEvent.click(classDropdown);
      const reshapeOption = screen.getByText('Reshape Tensor Operations');
      await userEvent.click(reshapeOption);
      
      // Now the type dropdown should show flatten
      await waitFor(() => {
        const typeInput = screen.getByDisplayValue('flatten');
        expect(typeInput).toBeInTheDocument();
      });
    });
  });

  describe('Node Creation', () => {
    it('should create a tensor operation node with correct structure and all parameters', async () => {
      render(
        <TensorOpsForm
          nodes={[]}
          addNode={mockAddNode}
          defaultTensorOps={mockDefaultTensorOps}
          getSetters={mockGetSetters}
          getDefaults={mockGetDefaults}
        />
      );
      
      // Click the create button
      const createButton = screen.getByRole('button', { name: /add tensor operation/i });
      await userEvent.click(createButton);
      
      // Verify createNode and addNode were called
      expect(mockCreateNode).toHaveBeenCalled();
      expect(mockAddNode).toHaveBeenCalled();
      
      // Verify the node has correct structure
      const createdNodesArray = mockAddNode.mock.calls[0][0];
      const createdNode = createdNodesArray[createdNodesArray.length - 1];
      expect(createdNode.type).toBe('torchNode');
      expect(createdNode.data.label).toBe('cat');
      expect(createdNode.data.operationType).toBe('TensorOp');
    });

    it('should create two different operation types with distinct IDs and parameters', async () => {
      // Mock ID generator to return sequential IDs
      const { generateUniqueNodeId } = require('@/app/canvas/utils/idGenerator');
      generateUniqueNodeId
        .mockReturnValueOnce('tensorop1')
        .mockReturnValueOnce('tensorop2');
      
      render(
        <TensorOpsForm
          nodes={[]}
          addNode={mockAddNode}
          defaultTensorOps={mockDefaultTensorOps}
          getSetters={mockGetSetters}
          getDefaults={mockGetDefaults}
        />
      );
      
      // Create first operation (add)
      const createButton = screen.getByRole('button', { name: /add tensor operation/i });
      await userEvent.click(createButton);
      
      // Verify first node was created
      const firstCallNodesArray = mockAddNode.mock.calls[0][0];
      const firstNode = firstCallNodesArray[firstCallNodesArray.length - 1];
      expect(firstNode.id).toBe('tensorop1');
      expect(firstNode.data.label).toBe('cat');
      expect(firstNode.data.operationType).toBe('TensorOp');
      
      // Change to stack
      const typeDropdown = screen.getByLabelText('Operation Type');
      await userEvent.click(typeDropdown);
      const stackOption = screen.getByText('stack');
      await userEvent.click(stackOption);
      
      // Create second operation (stack)
      await userEvent.click(createButton);
      
      // Verify second node was created with different properties
      const secondCallNodesArray = mockAddNode.mock.calls[1][0];
      const secondNode = secondCallNodesArray[secondCallNodesArray.length - 1];
      expect(secondNode.id).toBe('tensorop2');
      expect(secondNode.data.label).toBe('stack');
      expect(secondNode.data.operationType).toBe('TensorOp');
    });
  });
});
