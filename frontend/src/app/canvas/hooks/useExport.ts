export default function useExport(nodes: any[], edges: any[], defaultLayers: any[] = [], defaultTensorOps: any[] = [], defaultActivators: any[] = []) {
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
      alert("No input nodes found! Please add an input node to export.");
      return;
    }

    // Combine all default definitions for lookup
    const allDefaults = [
      ...defaultLayers,
      ...defaultTensorOps, 
      ...defaultActivators
    ];

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
      const exportNode = buildNodeFromReactFlowNode(currentNode, incomingEdges, allDefaults);
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

    const exportData = {
      nodes: exportNodes
    };

    // DEBUG: Show the generated JSON structure
    console.log("=== GENERATED JSON STRUCTURE ===");
    console.log(JSON.stringify(exportData, null, 2));
    console.log("=== END DEBUG ===");
    
    
    // Send JSON to backend API for Python conversion
    try {
      const response = await fetch('http://localhost:5000/convert-json-to-python', {
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
        
        alert("Python code generated successfully!");
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      console.error('Error converting JSON to Python:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      alert(`Failed to convert JSON to Python: ${errorMessage}`);
      
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
function buildNodeFromReactFlowNode(node: any, incomingEdges: Record<string, string[]>, allDefaults: any[]) {
  // Find the default definition for this node type
  const defaultDef = allDefaults.find(def => def.type === node.data.type);
  
  // Use operation type directly from node data
  const operation_type = node.data.operationType;

  // Determine parent(s) based on incoming edges
  const parents = incomingEdges[node.id] || [];
  let parent: string | string[] | null;
  
  if (parents.length === 0) {
    parent = null;
  } else if (parents.length === 1) {
    parent = parents[0];
  } else {
    parent = parents; // Multiple parents for operations like concat
  }

  // Use the type from the default definition or fall back to node data
  const type = defaultDef ? defaultDef.type : node.data.type;
  
  // Use the node's actual parameters
  const parameters = node.data.parameters || {};

  return {
    id: node.id,
    type: type,
    operation_type: operation_type,
    parent: parent,
    position: node.position || { x: 0, y: 0 },
    parameters: parameters
  };
}