// Utility function to generate unique IDs
export const generateUniqueId = (prefix: string = 'node'): string => {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

// Alternative function that checks existing nodes to ensure uniqueness
export const generateUniqueNodeId = (prefix: string, existingNodes: any[]): string => {
  const existingIds = new Set(existingNodes.map(node => node.id));
  let counter = 1;
  let newId = `${prefix}${counter}`;
  
  while (existingIds.has(newId)) {
    counter++;
    newId = `${prefix}${counter}`;
  }
  
  return newId;
};