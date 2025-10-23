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

    // BFS with dynamic branch tracking
interface BranchTracker {
  branchCounter: number;
  nodeToBranch: Record<string, string>;
  branchInputs: Record<string, Set<string>>;
}

const createBranchTracker = (): BranchTracker => ({
  branchCounter: 0,
  nodeToBranch: {},
  branchInputs: {}
});

const createNewBranch = (tracker: BranchTracker): string => {
  tracker.branchCounter += 1;
  return `branch_${tracker.branchCounter}`;
};

const detectConvergence = (
  nodeInputs: string[], 
  tracker: BranchTracker, 
  inputNodes: any[]
): { isConvergence: boolean; newBranch: string | null } => {
  if (nodeInputs.length <= 1) {
    return { isConvergence: false, newBranch: null };
  }

  // Get branches of all input nodes
  const inputBranches = new Set<string>();
  for (const inputRef of nodeInputs) {
    if (inputRef === "input") continue; // Skip input tensor
    
    // Find the source node from inputNodes or regular nodes
    const sourceNode = inputNodes.find(n => {
      const inputLabel = n.data.label || n.id;
      const inputName = inputLabel.replace("Input: ", "").trim() || "input";
      return inputName === inputRef;
    }) || nodes.find(n => `${n.id}_out` === inputRef);
    
    if (sourceNode) {
      const branch = tracker.nodeToBranch[sourceNode.id];
      if (branch) {
        inputBranches.add(branch);
      }
    }
  }

  // If inputs come from multiple branches, this is a convergence point
  if (inputBranches.size > 1) {
    const newBranch = createNewBranch(tracker);
    tracker.branchInputs[newBranch] = new Set(inputBranches);
    console.log(`Convergence detected: branches [${Array.from(inputBranches).join(', ')}] -> ${newBranch}`);
    return { isConvergence: true, newBranch };
  }

  return { isConvergence: false, newBranch: null };
};

// BFS traversal with branch tracking
const branchTracker = createBranchTracker();
const layerNodes = nodes.filter(node => node.data.operationType !== "Input");

// Build in-degree count for topological sort
const inDegree: Record<string, number> = {};
layerNodes.forEach(node => {
  inDegree[node.id] = (incomingEdges[node.id] || []).length;
});

// Initialize queue with nodes that have no dependencies or only depend on input
const queue: string[] = [];
layerNodes.forEach(node => {
  const deps = incomingEdges[node.id] || [];
  const nonInputDeps = deps.filter(dep => 
    !inputNodes.some(inp => inp.id === dep)
  );
  if (nonInputDeps.length === 0) {
    queue.push(node.id);
  }
});

const orderedLayers: any[] = [];
const processedNodes = new Set<string>();

while (queue.length > 0) {
  const currentId = queue.shift()!;
  if (processedNodes.has(currentId)) continue;

  const currentNode = nodes.find(n => n.id === currentId);
  if (!currentNode) continue;

  // Get node inputs for branch detection
  const nodeInputs = (incomingEdges[currentId] || []).map(sourceId => {
    // Check if source is an input node
    const inputNode = inputNodes.find(inp => inp.id === sourceId);
    if (inputNode) {
      const inputLabel = inputNode.data.label || inputNode.id;
      return inputLabel.replace("Input: ", "").trim() || "input";
    }
    // Otherwise it's a regular node output
    return `${sourceId}_out`;
  });

  // Detect convergence and assign branch
  const { isConvergence, newBranch } = detectConvergence(nodeInputs, branchTracker, inputNodes);
  
  if (isConvergence && newBranch) {
    // This node represents a convergence - assign to new branch
    branchTracker.nodeToBranch[currentId] = newBranch;
  } else {
    // Assign to same branch as first input (if any)
    if (nodeInputs.length > 0 && nodeInputs[0] !== "input") {
      // Find the source node
      const firstInputRef = nodeInputs[0];
      const sourceNode = nodes.find(n => `${n.id}_out` === firstInputRef);
      if (sourceNode) {
        const inputBranch = branchTracker.nodeToBranch[sourceNode.id];
        if (inputBranch) {
          branchTracker.nodeToBranch[currentId] = inputBranch;
        } else {
          // First node in a new branch
          const newBranch = createNewBranch(branchTracker);
          branchTracker.nodeToBranch[currentId] = newBranch;
        }
      } else {
        // Direct connection to input - create new branch
        const newBranch = createNewBranch(branchTracker);
        branchTracker.nodeToBranch[currentId] = newBranch;
      }
    } else {
      // Node directly connected to input or no inputs - create new branch
      const newBranch = createNewBranch(branchTracker);
      branchTracker.nodeToBranch[currentId] = newBranch;
    }
  }

  // Build layer object and add branch info
  const layer = buildLayerFromNode(currentNode, incomingEdges, inputNodes, allDefaults, nodes) as any;
  layer.branch = branchTracker.nodeToBranch[currentId];
  layer.operation_type = currentNode.data.operationType === "TensorOp" ? "tensor_op" : "layer";
  orderedLayers.push(layer);
  
  processedNodes.add(currentId);

  // Add dependent nodes to queue
  const dependents = outgoingEdges[currentId] || [];
  dependents.forEach(depId => {
    inDegree[depId] -= 1;
    if (inDegree[depId] === 0) {
      queue.push(depId);
    }
  });
}

// Find output nodes (nodes with no outgoing connections, excluding inputs)
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
      nodes: orderedLayers,  // Changed from "layers" to "nodes" for BFS compatibility
      outputs: outputs
    };

    // DEBUG: Show the generated JSON structure
    console.log("=== GENERATED JSON STRUCTURE ===");
    console.log(JSON.stringify(exportData, null, 2));
    console.log("=== END DEBUG ===");
    
    // Uncomment this line to see the JSON in an alert popup
    // alert("Generated JSON (check console for full structure):\n" + JSON.stringify(exportData, null, 2).substring(0, 500) + "...");

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