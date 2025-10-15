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

    // Find input nodes (nodes with operationType "Input")
    const inputNodes = nodes.filter(node => node.data.operationType === "Input");
    
    if (inputNodes.length === 0) {
      alert("No input nodes found! Please add an input node to export.");
      return;
    }

    // Create inputs array from input nodes
    const inputs: string[] = [];
    inputNodes.forEach(node => {
      const label = node.data.label || node.id;
      const params = node.data.parameters || {};
      
      // Extract input name
      const inputName = label.replace("Input: ", "").trim() || "input";
      inputs.push(inputName);
      
      // Add shape if available
      if (params.dims && Array.isArray(params.dims)) {
        let shapeStr = "";
        if (params.shapeType === "1D") {
          shapeStr = params.dims.join("");
        } else if (params.shapeType === "2D") {
          shapeStr = params.dims.join("x");
        } else if (params.shapeType === "3D") {
          shapeStr = params.dims.join("x");
        }
        if (shapeStr) {
          inputs.push(shapeStr);
        }
      }
    });

    // Combine all default definitions for lookup
    const allDefaults = [
      ...defaultLayers,
      ...defaultTensorOps, 
      ...defaultActivators
    ];

    // Traverse the graph starting from input nodes
    const visitedNodes = new Set<string>();
    const orderedLayers: any[] = [];
    
    // Function to traverse and build layers in order
    const traverseFromNode = (nodeId: string) => {
      if (visitedNodes.has(nodeId)) return;
      
      const node = nodes.find(n => n.id === nodeId);
      if (!node) return;
      
      visitedNodes.add(nodeId);
      
      // Skip input nodes in the layers array (they're handled in inputs)
      if (node.data.operationType !== "Input") {
        // Build layer object
        const layer = buildLayerFromNode(node, incomingEdges, inputNodes, allDefaults, nodes);
        orderedLayers.push(layer);
      }
      
      // Continue traversing to connected nodes
      const connectedNodes = outgoingEdges[nodeId] || [];
      connectedNodes.forEach(targetNodeId => {
        traverseFromNode(targetNodeId);
      });
    };

    // Start traversal from each input node
    inputNodes.forEach(inputNode => {
      traverseFromNode(inputNode.id);
    });

    // Find output nodes (nodes with no outgoing connections, excluding inputs)
    const layerNodes = nodes.filter(node => node.data.operationType !== "Input");
    const outputNodes = layerNodes.filter(node => 
      !outgoingEdges[node.id] || outgoingEdges[node.id].length === 0
    );
    
    const outputs = outputNodes.map(node => `${node.id}_out`);

    // Get unique library types used from the defaults
    const usedTypes = [...new Set(orderedLayers.map(layer => layer.type))];
    
    const exportData = {
      version: 1.0,
      libraries: {
        "torch.nn": usedTypes
      },
      inputs: inputs,
      layers: orderedLayers,
      outputs: outputs
    };

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

// Helper function to build a layer object from a node
function buildLayerFromNode(node: any, incomingEdges: Record<string, string[]>, inputNodes: any[], allDefaults: any[], allNodes: any[]) {
  // Find the default definition for this node type
  const defaultDef = allDefaults.find(def => def.type === node.data.type);
  
  // Use the type from the default definition (which should already have proper capitalization)
  const type = defaultDef ? defaultDef.type : node.data.type;
  
  // Get layer ID from label
  const [typeRaw, ...labelParts] = (node.data.label || "").split(":");
  const id = labelParts.join(":").trim() || node.id;
  
  // Build inputs array with proper input node detection
  const nodeInputs = (incomingEdges[node.id] || []).map((sourceId) => {
    // Find the source node in ALL nodes, not just inputNodes
    const sourceNode = allNodes.find(n => n.id === sourceId);
    
    if (sourceNode && sourceNode.data.operationType === "Input") {
      // This source is an Input node, use the input name
      const inputLabel = sourceNode.data.label || sourceNode.id;
      return inputLabel.replace("Input: ", "").trim() || "input";
    } else {
      // This source is a regular layer, use its output name
      return `${sourceId}_out`;
    }
  });
  
  // Remove duplicates in case there are multiple connections from the same source
  const uniqueInputs = [...new Set(nodeInputs)];
  
  // Use the node's actual parameters (which should be properly formatted already)
  const parameters = node.data.parameters || {};

  return {
    id,
    type: type,
    inputs: uniqueInputs,
    outputs: [`${node.id}_out`],
    parameters
  };
}