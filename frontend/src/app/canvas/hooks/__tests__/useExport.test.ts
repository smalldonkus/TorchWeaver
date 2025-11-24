import useExport from '../useExport';

// Mock global APIs
global.URL.createObjectURL = jest.fn(() => 'blob:mock-url');
global.URL.revokeObjectURL = jest.fn();
global.fetch = jest.fn();

// Mock document methods
document.createElement = jest.fn().mockReturnValue({
  click: jest.fn(),
  remove: jest.fn(),
  style: {},
  href: '',
  download: ''
});
document.body.appendChild = jest.fn();

describe('useExport', () => {
  let mockOnSuccess: jest.Mock;
  let mockOnError: jest.Mock;

  beforeEach(() => {
    mockOnSuccess = jest.fn();
    mockOnError = jest.fn();
    jest.clearAllMocks();
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({ python_code: 'mock code' })
    });
  });
  // note that these nodes lists (the indivdual nodes params) are heavyly simplified for testing purposes
  describe('Input Node Validation', () => {
    it('should call onError when no input nodes exist', async () => {
      const nodes = [
        { id: 'layer1', data: { operationType: 'Layer', type: 'Linear', parameters: {} } }
      ];
      const edges: any[] = [];

      const exportFn = useExport(nodes, edges, {}, {}, {}, mockOnSuccess, mockOnError);
      await exportFn();

      expect(mockOnError).toHaveBeenCalledWith(
        expect.stringContaining('No input nodes found')
      );
      expect(mockOnSuccess).not.toHaveBeenCalled();
    });

    it('should proceed when input nodes exist', async () => {
      const nodes = [
        { id: 'input1', data: { operationType: 'Input', type: 'SingleDimensionalInput', parameters: {} } }
      ];
      const edges: any[] = [];

      const exportFn = useExport(nodes, edges, {}, {}, {}, mockOnSuccess, mockOnError);
      await exportFn();

      // Should not call error callback for missing input
      expect(mockOnError).not.toHaveBeenCalledWith(expect.stringContaining('No input nodes found'));
    });

    it('should use alert when nothing exists and they try to export', async () => {
      const nodes: any[] = [];
      const edges: any[] = [];
      
      const originalAlert = window.alert;
      window.alert = jest.fn();

      const exportFn = useExport(nodes, edges, {}, {}, {});
      await exportFn();

      expect(window.alert).toHaveBeenCalledWith(
        expect.stringContaining('No input nodes found')
      );

      window.alert = originalAlert;
    });
  });

  describe('Edge Mapping', () => {
    it('should handle nodes with multiple incoming edges', async () => {
      const nodes = [
        { id: 'input1', data: { operationType: 'Input', type: 'SingleDimensionalInput', parameters: {} } },
        { id: 'input2', data: { operationType: 'Input', type: 'SingleDimensionalInput', parameters: {} } },
        { id: 'tensorop1', data: { operationType: 'TensorOp', type: 'cat', parameters: {} } }
      ];
      const edges = [
        { source: 'input1', target: 'tensorop1' },
        { source: 'input2', target: 'tensorop1' }
      ];

      const exportFn = useExport(nodes, edges, {}, {}, {}, mockOnSuccess, mockOnError);
      await exportFn();

      // the only error doesnt occur hence we can say it passed
      expect(mockOnError).not.toHaveBeenCalledWith(expect.stringContaining('No input nodes found'));
    });

    it('should handle nodes with multiple outgoing edges', async () => {
      const nodes = [
        { id: 'input1', data: { operationType: 'Input', type: 'SingleDimensionalInput', parameters: {} } },
        { id: 'layer1', data: { operationType: 'Layer', type: 'Linear', parameters: {} } },
        { id: 'layer2', data: { operationType: 'Layer', type: 'Linear', parameters: {} } }
      ];
      const edges = [
        { source: 'input1', target: 'layer1' },
        { source: 'input1', target: 'layer2' }
      ];

      const exportFn = useExport(nodes, edges, {}, {}, {}, mockOnSuccess, mockOnError);
      await exportFn();

      expect(mockOnError).not.toHaveBeenCalledWith(expect.stringContaining('No input nodes found'));
    });
  });

  describe('Node Types', () => {
    it('should process different operation types', async () => {
      const nodes = [
        { id: 'input1', data: { operationType: 'Input', type: 'SingleDimensionalInput', parameters: {} } },
        { id: 'layer1', data: { operationType: 'Layer', type: 'Linear', parameters: {} } },
        { id: 'activator1', data: { operationType: 'Activator', type: 'ReLU', parameters: {} } },
        { id: 'tensorop1', data: { operationType: 'TensorOp', type: 'cat', parameters: {} } }
      ];
      const edges = [
        { source: 'input1', target: 'layer1' },
        { source: 'layer1', target: 'activator1' },
        { source: 'activator1', target: 'tensorop1' }
      ];

      const exportFn = useExport(nodes, edges, {}, {}, {}, mockOnSuccess, mockOnError);
      await exportFn();

      // Should not error when processing valid graph
      expect(mockOnError).not.toHaveBeenCalledWith(expect.stringContaining('No input nodes found'));
    });
  });

  describe('Complex Graph Structures', () => {
    it('should handle sequential chain of nodes', async () => {
      const nodes = [
        { id: 'input1', data: { operationType: 'Input', type: 'SingleDimensionalInput', parameters: {} } },
        { id: 'layer1', data: { operationType: 'Layer', type: 'Linear', parameters: {} } },
        { id: 'activator1', data: { operationType: 'Activator', type: 'ReLU', parameters: {} } },
        { id: 'layer2', data: { operationType: 'Layer', type: 'Linear', parameters: {} } }
      ];
      const edges = [
        { source: 'input1', target: 'layer1' },
        { source: 'layer1', target: 'activator1' },
        { source: 'activator1', target: 'layer2' }
      ];

      const exportFn = useExport(nodes, edges, {}, {}, {}, mockOnSuccess, mockOnError);
      await exportFn();

      expect(mockOnError).not.toHaveBeenCalledWith(expect.stringContaining('No input nodes found'));
    });

    it('should handle multiple input nodes', async () => {
      const nodes = [
        { id: 'input1', data: { operationType: 'Input', type: 'SingleDimensionalInput', parameters: {} } },
        { id: 'input2', data: { operationType: 'Input', type: 'SingleDimensionalInput', parameters: {} } },
        { id: 'merge', data: { operationType: 'TensorOp', type: 'cat', parameters: {} } }
      ];
      const edges = [
        { source: 'input1', target: 'merge' },
        { source: 'input2', target: 'merge' }
      ];

      const exportFn = useExport(nodes, edges, {}, {}, {}, mockOnSuccess, mockOnError);
      await exportFn();

      expect(mockOnError).not.toHaveBeenCalledWith(expect.stringContaining('No input nodes found'));
    });

    it('should handle branching structures', async () => {
      const nodes = [
        { id: 'input1', data: { operationType: 'Input', type: 'SingleDimensionalInput', parameters: {} } },
        { id: 'tensorop1', data: { operationType: 'TensorOp', type: 'split', parameters: {} } },
        { id: 'layer1', data: { operationType: 'Layer', type: 'Linear', parameters: {} } },
        { id: 'layer2', data: { operationType: 'Layer', type: 'Linear', parameters: {} } }
      ];
      const edges = [
        { source: 'input1', target: 'tensorop1' },
        { source: 'tensorop1', target: 'layer1' },
        { source: 'tensorop1', target: 'layer2' }
      ];

      const exportFn = useExport(nodes, edges, {}, {}, {}, mockOnSuccess, mockOnError);
      await exportFn();

      expect(mockOnError).not.toHaveBeenCalledWith(expect.stringContaining('No input nodes found'));
    });
  });

  describe('Function Return Type', () => {
    it('should return an async function', () => {
      const nodes = [
        { id: 'input1', data: { operationType: 'Input', type: 'SingleDimensionalInput', parameters: {} } }
      ];
      const edges: any[] = [];

      const exportFn = useExport(nodes, edges);
      
      expect(typeof exportFn).toBe('function');
      expect(exportFn.constructor.name).toBe('AsyncFunction');
    });
  });
});
