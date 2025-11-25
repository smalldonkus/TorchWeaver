import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';
import InputForm from '../InputForm';

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

describe('InputForm', () => {
  const mockAddNode = jest.fn();
  const mockGetSetters = jest.fn(() => ({}));
  const mockGetDefaults = jest.fn(() => ({}));

    // Note we have simplified default layers for testing
    // or created ones that dont exist in our database currently
    // to cover various input scenarios
    // such as ImageInput
  const mockDefaultInputs = {
    data: {
      'SingleDimensionalInput': {
        'SingleDimensionalInput (no batch)': {
          library: null,
          parameters: {
            features: 1
          },
          parameters_format: {
            features: ['int']
          }
        },
        'SingleDimensionalInput (with batch)': {
          library: null,
          parameters: {
            batch_size: 1,
            features: 1
          },
          parameters_format: {
            batch_size: ['int'],
            features: ['int']
          }
        }
      },
      'ImageInput': {
        'ImageInput (no batch)': {
          library: null,
          parameters: {
            channels: 3,
            height: 224,
            width: 224
          },
          parameters_format: {
            channels: ['int'],
            height: ['int'],
            width: ['int']
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
    it('should update type dropdown when category is changed', async () => {
      render(
        <InputForm
          nodes={[]}
          addNode={mockAddNode}
          defaultInputs={mockDefaultInputs}
          getSetters={mockGetSetters}
          getDefaults={mockGetDefaults}
        />
      );
      
      // Verify initial state - should show SingleDimensionalInput (no batch)
      await waitFor(() => {
        const typeInput = screen.getByDisplayValue('SingleDimensionalInput (no batch)');
        expect(typeInput).toBeInTheDocument();
      });
      
      // Change category to ImageInput
      const categoryDropdown = screen.getByLabelText('Input Category');
      await userEvent.click(categoryDropdown);
      const imageInputOption = screen.getByText('ImageInput');
      await userEvent.click(imageInputOption);
      
      // Now the type dropdown should show ImageInput (no batch)
      await waitFor(() => {
        const typeInput = screen.getByDisplayValue('ImageInput (no batch)');
        expect(typeInput).toBeInTheDocument();
      });
    });
  });

  describe('Node Creation', () => {
    it('should create an input node with correct structure and all parameters', async () => {
      render(
        <InputForm
          nodes={[]}
          addNode={mockAddNode}
          defaultInputs={mockDefaultInputs}
          getSetters={mockGetSetters}
          getDefaults={mockGetDefaults}
        />
      );
      
      // Click the create button
      const createButton = screen.getByRole('button', { name: /add input/i });
      await userEvent.click(createButton);
      
      // Verify createNode and addNode were called
      expect(mockCreateNode).toHaveBeenCalled();
      expect(mockAddNode).toHaveBeenCalled();
      
      // Verify the node has correct structure
      const createdNodesArray = mockAddNode.mock.calls[0][0];
      const createdNode = createdNodesArray[createdNodesArray.length - 1];
      expect(createdNode.type).toBe('torchNode');
      expect(createdNode.data.label).toBe('SingleDimensionalInput (no batch)');
      expect(createdNode.data.operationType).toBe('Input');
    });

    it('should create two different input types with distinct IDs and parameters', async () => {
      // Mock ID generator to return sequential IDs
      const { generateUniqueNodeId } = require('@/app/canvas/utils/idGenerator');
      generateUniqueNodeId
        .mockReturnValueOnce('input1')
        .mockReturnValueOnce('input2');
      
      render(
        <InputForm
          nodes={[]}
          addNode={mockAddNode}
          defaultInputs={mockDefaultInputs}
          getSetters={mockGetSetters}
          getDefaults={mockGetDefaults}
        />
      );
      
      // Create first input (Input)
      const createButton = screen.getByRole('button', { name: /add input/i });
      await userEvent.click(createButton);
      
      // Verify first node was created
      const firstCallNodesArray = mockAddNode.mock.calls[0][0];
      const firstNode = firstCallNodesArray[firstCallNodesArray.length - 1];
      expect(firstNode.id).toBe('input1');
      expect(firstNode.data.label).toBe('SingleDimensionalInput (no batch)');
      expect(firstNode.data.operationType).toBe('Input');
      
      // Change to SingleDimensionalInput (with batch)
      const typeDropdown = screen.getByLabelText('Input Type');
      await userEvent.click(typeDropdown);
      const withBatchOption = screen.getByText('SingleDimensionalInput (with batch)');
      await userEvent.click(withBatchOption);
      
      // Create second input
      await userEvent.click(createButton);
      
      // Verify second node was created with different properties
      const secondCallNodesArray = mockAddNode.mock.calls[1][0];
      const secondNode = secondCallNodesArray[secondCallNodesArray.length - 1];
      expect(secondNode.id).toBe('input2');
      expect(secondNode.data.label).toBe('SingleDimensionalInput (with batch)');
      expect(secondNode.data.operationType).toBe('Input');
    });
  });
});
