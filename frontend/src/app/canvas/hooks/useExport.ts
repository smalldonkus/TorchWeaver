export default function useExport(
  nodes: any[], 
  edges: any[], 
  defaultLayers: any = {}, 
  defaultTensorOps: any = {}, 
  defaultActivators: any = {},
  onSuccess?: (message: string) => void,
  onError?: (message: string) => void
) {
  return async () => {
    // Build adjacency maps for traversal
    const outgoingEdges: Record<string, string[]> = {};
    const incomingEdges: Record<string, string[]> = {};
    
    edges.forEach((edge) => {
      // Map: nodeId -> [list of nodes it connects to]
      if (!outgoingEdges[edge.source]) outgoingEdges[edge.source] = [];
      outgoingEdges[edge.source].push(edge.target);
      
      // Map: nodeId -> [list of nodes that connect to it]
      if (!incomingEdges[edge.target]) incomingEdges[edge.target] = [];
      incomingEdges[edge.target].push(edge.source);
    });

    // Find input nodes
    const inputNodes = nodes.filter(node => node.data.operationType === "Input");
    
    if (inputNodes.length === 0) {
      const message = "No input nodes found! Please add an input node to export.";
      if (onError) {
        onError(message);
      } else {
        alert(message);
      }
      return;
    }

    // Helper function to find a type definition in hierarchical structure
    const findTypeDefinition = (data: any, targetType: string): any => {
      if (!data || !data.data) return null;
      
      for (const [className, classItems] of Object.entries(data.data)) {
        const typedClassItems = classItems as Record<string, any>;
        if (typedClassItems[targetType]) {
          return {
            type: targetType,
            class: className,
            ...typedClassItems[targetType]
          };
        }
      }
      return null;
    };

    // Helper function to find a type definition across all operation types
    const findAnyTypeDefinition = (targetType: string): any => {
      return findTypeDefinition(defaultLayers, targetType) ||
             findTypeDefinition(defaultTensorOps, targetType) ||
             findTypeDefinition(defaultActivators, targetType);
    };

    // Helper function to check if a node is a split operation
    const isSplitOperation = (nodeId: string): boolean => {
      const node = nodes.find(n => n.id === nodeId);
      if (!node || node.data.operationType !== "TensorOp") return false;
      
      const tensorOpDef = findTypeDefinition(defaultTensorOps, node.data.type);
      return tensorOpDef?.codeGeneration?.operationPattern === "split";
    };

    // Helper function to get split output name for a child of a split operation
    const getSplitOutputName = (splitNodeId: string, childIndex: number): string => {
      const suffixes = ['a', 'b', 'c', 'd', 'e', 'f']; // Support up to 6 outputs
      return `${splitNodeId}${suffixes[childIndex] || childIndex}`;
    };

    // Build the new JSON structure with all nodes and parent relationships
    const exportNodes: any[] = [];
    const processedNodes = new Set<string>();

    // Start traversal from input nodes and build in correct order
    const queue: string[] = [];
    
    // Add all input nodes to queue first
    inputNodes.forEach(inputNode => {
      queue.push(inputNode.id);
    });

    // Process nodes in topological order starting from inputs
    while (queue.length > 0) {
      const currentNodeId = queue.shift()!;
      
      if (processedNodes.has(currentNodeId)) continue;
      
      const currentNode = nodes.find(n => n.id === currentNodeId);
      if (!currentNode) continue;

      // Check if all parents have been processed
      const parents = incomingEdges[currentNodeId] || [];
      const allParentsProcessed = parents.every(parentId => processedNodes.has(parentId));
      
      if (!allParentsProcessed) {
        // Put back at end of queue and continue
        queue.push(currentNodeId);
        continue;
      }

      // Process this node
      const exportNode = buildNodeFromReactFlowNode(currentNode, incomingEdges, outgoingEdges, findAnyTypeDefinition, isSplitOperation, getSplitOutputName);
      exportNodes.push(exportNode);
      processedNodes.add(currentNodeId);

      // Add children to queue
      const children = outgoingEdges[currentNodeId] || [];
      children.forEach(childId => {
        if (!processedNodes.has(childId)) {
          queue.push(childId);
        }
      });
    }

    // Post-process: Update parent references for children of split operations
    exportNodes.forEach(node => {
      if (node.parent && typeof node.parent === 'string') {
        const parentNode = exportNodes.find(n => n.id === node.parent);
        if (parentNode && isSplitOperation(parentNode.id)) {
          // Find the index of this node in the parent's children array
          const childIndex = parentNode.children.indexOf(node.id);
          if (childIndex >= 0) {
            node.parent = getSplitOutputName(parentNode.id, childIndex);
          }
        }
      } else if (Array.isArray(node.parent)) {
        // Handle multiple parents (for concat operations)
        node.parent = node.parent.map((parentId: string) => {
          const parentNode = exportNodes.find(n => n.id === parentId);
          if (parentNode && isSplitOperation(parentNode.id)) {
            const childIndex = parentNode.children.indexOf(node.id);
            return childIndex >= 0 ? getSplitOutputName(parentNode.id, childIndex) : parentId;
          }
          return parentId;
        });
      }
    });

    const exportData = {
      nodes: exportNodes
    };

    // DEBUG: Show the generated JSON structure
    console.log("=== GENERATED JSON STRUCTURE ===");
    console.log(JSON.stringify(exportData, null, 2));
    console.log("=== END DEBUG ===");
    
    
    // Send JSON to backend API for Python code generation
    try {
      const response = await fetch('http://localhost:5000/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(exportData)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        // Download the Python code
        const blob = new Blob([result.python_code], { type: "text/plain" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "model.py";
        link.click();
        URL.revokeObjectURL(url);
        
        const successMessage = "Python code generated successfully!";
        if (onSuccess) {
          onSuccess(successMessage);
        } else {
          alert(successMessage);
        }
      } else {
        const errorMsg = `Error: ${result.error}`;
        if (onError) {
          onError(errorMsg);
        } else {
          alert(errorMsg);
        }
      }
    } catch (error) {
      console.error('Error converting JSON to Python:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      const fullErrorMsg = `Failed to convert JSON to Python: ${errorMessage}`;
      if (onError) {
        onError(fullErrorMsg);
      } else {
        alert(fullErrorMsg);
      }
      
      // Fallback: download JSON if API fails
      const json = JSON.stringify(exportData, null, 2);
      const blob = new Blob([json], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "model.json";
      link.click();
      URL.revokeObjectURL(url);
    }
  };
}

// Helper function to build a node object from a ReactFlow node using the new parent structure
function buildNodeFromReactFlowNode(
  node: any, 
  incomingEdges: Record<string, string[]>, 
  outgoingEdges: Record<string, string[]>, 
  findDefinition: (targetType: string) => any,
  isSplitOperation: (nodeId: string) => boolean,
  getSplitOutputName: (splitNodeId: string, childIndex: number) => string
) {
  // Find the default definition for this node type
  const defaultDef = findDefinition(node.data.type);
  
  // Use operation type directly from node data
  const operation_type = node.data.operationType;

  // Determine parent(s) based on incoming edges (simple approach - we'll fix split references later)
  const parents = incomingEdges[node.id] || [];
  let parent: string | string[] | null;
  
  if (parents.length === 0) {
    parent = null;
  } else if (parents.length === 1) {
    parent = parents[0];
  } else {
    parent = parents; // Multiple parents for operations like concat
  }

  // Calculate outgoing edges count
  const outgoingEdgesCount = (outgoingEdges[node.id] || []).length;
  
  // Get children for this node
  const children = outgoingEdges[node.id] || [];

  // Use the type from the default definition or fall back to node data
  const type = defaultDef ? defaultDef.type : node.data.type;
  
  // Use the node's actual parameters, filtering out inherit_from_parent (frontend-only)
  const parameters = node.data.parameters ? 
    Object.fromEntries(
      Object.entries(node.data.parameters).filter(([key]) => key !== 'inherit_from_parent')
    ) : 
    {};

  return {
    id: node.id,
    type: type,
    operation_type: operation_type,
    parent: parent,
    children: children,
    outgoing_edges_count: outgoingEdgesCount,
    position: node.position || { x: 0, y: 0 },
    parameters: parameters
  };
}