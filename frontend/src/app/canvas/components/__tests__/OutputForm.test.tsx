import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';
import OutputForm from '../OutputForm';

// Mock the ID generator to return predictable IDs
jest.mock('@/app/canvas/utils/idGenerator', () => ({
  generateUniqueNodeId: jest.fn((prefix) => `${prefix}1`),
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

describe('OutputForm', () => {
  const mockAddNode = jest.fn();
  const mockGetSetters = jest.fn(() => ({}));
  const mockGetDefaults = jest.fn(() => ({}));

  beforeEach(() => {
    jest.clearAllMocks();
    mockCreateNode.mockClear();
  });

  it('should render the Add Output button', () => {
    render(
      <OutputForm
        nodes={[]}
        addNode={mockAddNode}
        getSetters={mockGetSetters}
        getDefaults={mockGetDefaults}
      />
    );

    const button = screen.getByRole('button', { name: /add output/i });
    expect(button).toBeInTheDocument();
  });

  it('should create an output node with correct ID format', async () => {
    render(
      <OutputForm
        nodes={[]}
        addNode={mockAddNode}
        getSetters={mockGetSetters}
        getDefaults={mockGetDefaults}
      />
    );

    const button = screen.getByRole('button', { name: /add output/i });
    await userEvent.click(button);

    // Verify createNode was called with output1
    expect(mockCreateNode).toHaveBeenCalledWith(
      'output1',
      0, // posModifier based on nodes.length
      'Output',
      'Output',
      'Output',
      {},
      mockGetSetters,
      mockGetDefaults
    );
  });

  it('should add output node to canvas with correct structure', async () => {
    render(
      <OutputForm
        nodes={[]}
        addNode={mockAddNode}
        getSetters={mockGetSetters}
        getDefaults={mockGetDefaults}
      />
    );

    const button = screen.getByRole('button', { name: /add output/i });
    await userEvent.click(button);

    expect(mockAddNode).toHaveBeenCalledTimes(1);
    
    const addedNodesArray = mockAddNode.mock.calls[0][0];
    const outputNode = addedNodesArray[addedNodesArray.length - 1];
    
    expect(outputNode.id).toBe('output1');
    expect(outputNode.type).toBe('torchNode');
    expect(outputNode.data.label).toBe('Output');

  });

// note while this can occur for outputs, they should be exporting with only one output
// we keep the test to ensure unique ID generation logic is sound

  it('should generate unique IDs when nodes already exist', async () => {
    const { generateUniqueNodeId } = require('@/app/canvas/utils/idGenerator');
    generateUniqueNodeId
      .mockReturnValueOnce('output1')
      .mockReturnValueOnce('output2');

    const existingNodes = [{ id: 'input1' }];

    const { rerender } = render(
      <OutputForm
        nodes={existingNodes}
        addNode={mockAddNode}
        getSetters={mockGetSetters}
        getDefaults={mockGetDefaults}
      />
    );

    const button = screen.getByRole('button', { name: /add output/i });
    await userEvent.click(button);

    expect(mockAddNode).toHaveBeenCalledTimes(1);
    let addedNodesArray = mockAddNode.mock.calls[0][0];
    expect(addedNodesArray[addedNodesArray.length - 1].id).toBe('output1');

    // Add second output
    const updatedNodes = [...existingNodes, addedNodesArray[addedNodesArray.length - 1]];
    
    rerender(
      <OutputForm
        nodes={updatedNodes}
        addNode={mockAddNode}
        getSetters={mockGetSetters}
        getDefaults={mockGetDefaults}
      />
    );

    await userEvent.click(button);

    expect(mockAddNode).toHaveBeenCalledTimes(2);
    addedNodesArray = mockAddNode.mock.calls[1][0];
    expect(addedNodesArray[addedNodesArray.length - 1].id).toBe('output2');
  });

  it('should pass empty parameters object for output nodes', async () => {
    render(
      <OutputForm
        nodes={[]}
        addNode={mockAddNode}
        getSetters={mockGetSetters}
        getDefaults={mockGetDefaults}
      />
    );

    const button = screen.getByRole('button', { name: /add output/i });
    await userEvent.click(button);

    const addedNodesArray = mockAddNode.mock.calls[0][0];
    const outputNode = addedNodesArray[addedNodesArray.length - 1];
    
    expect(outputNode.data.parameters).toEqual({});
  });
});
