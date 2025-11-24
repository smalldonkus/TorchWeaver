import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';
import ActivatorsForm from '../ActivatorsForm';

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

// Mock createNode
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

describe('ActivatorsForm', () => {
  const mockAddNode = jest.fn();
  const mockGetSetters = jest.fn(() => ({}));
  const mockGetDefaults = jest.fn(() => ({}));

  const mockDefaultActivators = {
    data: {
      'Activation Functions': {
        'ReLU': {
          library: 'torch.nn',
          parameters: {},
          parameters_format: {}
        },
        'Sigmoid': {
          library: 'torch.nn',
          parameters: {},
          parameters_format: {}
        },
        'ELU': {
          library: 'torch.nn',
          parameters: {
            alpha: 1.0
          },
          parameters_format: {
            alpha: ['float']
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

  // testing whether the dropdowns update correctly
  describe('UI Behavior', () => {
    it('should update type dropdown when class is changed', async () => {
      render(
        <ActivatorsForm
          nodes={[]}
          addNode={mockAddNode}
          defaultActivators={mockDefaultActivators}
          getSetters={mockGetSetters}
          getDefaults={mockGetDefaults}
        />
      );
      
      // The initial state should show ReLU from Activation Functions and this
      // is just to confirm the test setup is correct
      await waitFor(() => {
        const typeInput = screen.getByDisplayValue('ReLU');
        expect(typeInput).toBeInTheDocument();
      });
      
      // Change type to Sigmoid
      const typeDropdown = screen.getByLabelText('Activation Function Type');
      await userEvent.click(typeDropdown);
      const sigmoidOption = screen.getByText('Sigmoid');
      await userEvent.click(sigmoidOption);
      
      // Now the type dropdown should show Sigmoid
      await waitFor(() => {
        const typeInput = screen.getByDisplayValue('Sigmoid');
        expect(typeInput).toBeInTheDocument();
      });
    });
  });

  describe('Node Creation', () => {
    it('should create an activator node with correct structure and all parameters', async () => {
      render(
        <ActivatorsForm
          nodes={[]}
          addNode={mockAddNode}
          defaultActivators={mockDefaultActivators}
          getSetters={mockGetSetters}
          getDefaults={mockGetDefaults}
        />
      );
      
      const addButton = screen.getByRole('button', { name: /Add Activation Function/i });
      await userEvent.click(addButton);
      
      // Verify createNode and addNode were called
      expect(mockCreateNode).toHaveBeenCalled();
      expect(mockAddNode).toHaveBeenCalled();
      
      // Verify the node has correct structure with

      // operationType as 'Activator'
      // label as 'ReLU'
      // and type as 'torchNode' (note this is the node type, not the data type)

      const createdNodesArray = mockAddNode.mock.calls[0][0];
      const createdNode = createdNodesArray[createdNodesArray.length - 1];
      expect(createdNode.type).toBe('torchNode');
      expect(createdNode.data.label).toBe('ReLU');
      expect(createdNode.data.operationType).toBe('Activator');
    });
  });

  describe('Creating Multiple Activators', () => {
    it('should create two different activator types with distinct IDs and parameters', async () => {
      // Mock ID generator to return sequential IDs
      const { generateUniqueNodeId } = require('@/app/canvas/utils/idGenerator');
      generateUniqueNodeId
        .mockReturnValueOnce('activator1')
        .mockReturnValueOnce('activator2');
      
      render(
        <ActivatorsForm
          nodes={[]}
          addNode={mockAddNode}
          defaultActivators={mockDefaultActivators}
          getSetters={mockGetSetters}
          getDefaults={mockGetDefaults}
        />
      );
      
      // Add first activator (ReLU)
      const addButton = screen.getByRole('button', { name: /Add Activation Function/i });
      await userEvent.click(addButton);
      
      // Verify first node was created
      // same as before
      const firstCallNodes = mockAddNode.mock.calls[0][0];
      expect(firstCallNodes[0].id).toBe('activator1');
      expect(firstCallNodes[0].data.type).toBe('ReLU');
      expect(firstCallNodes[0].data.operationType).toBe('Activator');
      
      // Change to ELU and add second activator
  
      const typeDropdown = screen.getByLabelText('Activation Function Type');
      await userEvent.click(typeDropdown);
      const eluOption = screen.getByText('ELU');
      await userEvent.click(eluOption);
      await userEvent.click(addButton);
      
      // Verify both nodes have distinct IDs and different types
      expect(mockAddNode).toHaveBeenCalledTimes(2);
      expect(mockCreateNode).toHaveBeenCalledTimes(2);
      
      // Second node: ELU (different type)
      const secondCallNodes = mockAddNode.mock.calls[1][0];
      expect(secondCallNodes[0].id).toBe('activator2');
      expect(secondCallNodes[0].data.type).toBe('ELU');
      expect(secondCallNodes[0].data.operationType).toBe('Activator');
      
      // Verify both nodes were created independently with distinct IDs
      expect(firstCallNodes[0].id).not.toBe(secondCallNodes[0].id);
    });
  });
});
