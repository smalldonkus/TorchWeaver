import { generateUniqueId, generateUniqueNodeId } from '../idGenerator';

describe('idGenerator', () => {
  describe('generateUniqueId', () => {
    it('should generate an id with default prefix', () => {
      const id = generateUniqueId();
      expect(id).toMatch(/^node_\d+_[a-z0-9]+$/);
    });

    it('should generate an id with custom prefix', () => {
      const id = generateUniqueId('layer');
      expect(id).toMatch(/^layer_\d+_[a-z0-9]+$/);
    });

    it('should generate unique ids on multiple calls', () => {
      const id1 = generateUniqueId();
      const id2 = generateUniqueId();
      expect(id1).not.toBe(id2);
    });

    it('should include timestamp in id', () => {
      const beforeTime = Date.now();
      const id = generateUniqueId();
      const afterTime = Date.now();
      
      const timestamp = parseInt(id.split('_')[1]);
      expect(timestamp).toBeGreaterThanOrEqual(beforeTime);
      expect(timestamp).toBeLessThanOrEqual(afterTime);
    });
  });

  describe('generateUniqueNodeId', () => {
    it('should generate id with counter starting at 1', () => {
      const existingNodes: any[] = [];
      const id = generateUniqueNodeId('layer', existingNodes);
      expect(id).toBe('layer1');
    });

    it('should avoid existing ids', () => {
      const existingNodes = [
        { id: 'layer1' },
        { id: 'layer2' }
      ];
      const id = generateUniqueNodeId('layer', existingNodes);
      expect(id).toBe('layer3');
    });

    it('should handle gaps in existing ids', () => {
      const existingNodes = [
        { id: 'layer1' },
        { id: 'layer3' }
      ];
      const id = generateUniqueNodeId('layer', existingNodes);
      expect(id).toBe('layer2');
    });

    it('should work with different prefixes', () => {
      const existingNodes = [
        { id: 'layer1' },
        { id: 'activator1' }
      ];
      const id = generateUniqueNodeId('activator', existingNodes);
      expect(id).toBe('activator2');
    });

    it('should handle empty nodes array', () => {
      const id = generateUniqueNodeId('input', []);
      expect(id).toBe('input1');
    });
  });
});
