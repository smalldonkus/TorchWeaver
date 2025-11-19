// Channel propagation utilities for neural network node inheritance

// Helper function to get max output channels from all parent nodes
const getMaxParentOutputChannels = (nodeId: string, currentNodes: any[], currentEdges: any[]): number | null => {
  // Find all parent edges (edges pointing to this node)
  const parentEdges = currentEdges.filter(edge => edge.target === nodeId);
  
  if (parentEdges.length === 0) return null;
  
  // Get all parent nodes' output channels
  const parentOutputChannels = parentEdges
    .map(edge => {
      const parentNode = currentNodes.find(node => node.id === edge.source);
      return parentNode?.data.outputChannels;
    })
    .filter(channels => channels !== undefined && channels !== null);
  
  if (parentOutputChannels.length === 0) return null;
  
  // Return the maximum
  return Math.max(...parentOutputChannels);
};

// Helper function to find node definition based on operation type and specific type
export const findNodeDefinition = (
  operationType: string, 
  specificType: string,
  defaultLayers: any,
  defaultTensorOps: any,
  defaultActivators: any,
  defaultInputs: any
) => {
  let dataSource;
  switch (operationType) {
    case "Layer":
      dataSource = defaultLayers;
      break;
    case "TensorOp":
      dataSource = defaultTensorOps;
      break;
    case "Activator":
      dataSource = defaultActivators;
      break;
    case "Input":
      dataSource = defaultInputs;
      break;
    default:
      return null;
  }
  
  if (!dataSource?.data) {
    return null;
  }
  
  // Search through all classes to find the specific type
  for (const [className, classItems] of Object.entries(dataSource.data)) {
    if ((classItems as any)[specificType]) {
      return (classItems as any)[specificType];
    }
  }
  return null;
};

// Recursive function to propagate channel inheritance changes
export const propagateChannelInheritance = (
  parentNodeId: string,
  parentOutputChannels: number,
  currentNodes: any[],
  currentEdges: any[],
  defaultLayers: any,
  defaultTensorOps: any,
  defaultActivators: any,
  defaultInputs: any
): any[] => {
  // console.log(`🔄 Starting propagation from node ${parentNodeId} with ${parentOutputChannels} output channels`);
  
  // Find all direct children of the parent node
  const childrenIds = currentEdges
    .filter(edge => edge.source === parentNodeId)
    .map(edge => edge.target);
    
  let updatedNodes = [...currentNodes];
  
  // Process each child
  childrenIds.forEach(childId => {
    const childIndex = updatedNodes.findIndex(node => node.id === childId);
    if (childIndex === -1) {
      return;
    }
    
    const childNode = updatedNodes[childIndex];
    
    // Check if this child inherits from parent
    if (!childNode.data.can_inherit_from_parent) {
      return;
    }
    
    // Check if we should also update output channels
    const childDefinition = findNodeDefinition(
      childNode.data.operationType, 
      childNode.data.type,
      defaultLayers,
      defaultTensorOps,
      defaultActivators,
      defaultInputs
    );
    const canEditChannels = childDefinition?.parseCheck?.CanEditChannels;
    
    let updatedChildData = { ...childNode.data };
    let childOutputChannelsChanged = false;
    let newChildOutputChannels = childNode.data.outputChannels;
    
    if (!canEditChannels) {
      // Pass-through node with potentially multiple parents: use max output channels
      const maxParentChannels = getMaxParentOutputChannels(childId, updatedNodes, currentEdges);
      if (maxParentChannels !== null) {
        updatedChildData.inputChannels = maxParentChannels;
        updatedChildData.outputChannels = maxParentChannels;
        newChildOutputChannels = maxParentChannels;
        childOutputChannelsChanged = (childNode.data.outputChannels !== maxParentChannels);
      }
    } else {
      // Node can edit channels, update linked parameters
      const channelLinks = childDefinition?.parseCheck?.ChannelLinks || [];
      let updatedParameters = { ...childNode.data.parameters };
      let inputParam = null;
      let outputParam = null;
      
      channelLinks.forEach((link: any) => {
        if (link.inputParam) {
          inputParam = link.inputParam;
          updatedParameters[link.inputParam] = parentOutputChannels;
        }
        if (link.outputParam) {
          outputParam = link.outputParam;
        }
      });
      
      updatedChildData.parameters = updatedParameters;
      
      // If input and output parameters are linked to the same parameter, update output channels too
      if (inputParam && outputParam && inputParam === outputParam) {
        updatedChildData.outputChannels = parentOutputChannels;
        newChildOutputChannels = parentOutputChannels;
        childOutputChannelsChanged = (childNode.data.outputChannels !== parentOutputChannels);
      }
    }
    
    // Update the child node
    updatedNodes[childIndex] = {
      ...childNode,
      data: updatedChildData
    };
    
    // If child's output channels changed, recursively propagate to its children
    if (childOutputChannelsChanged) {
      updatedNodes = propagateChannelInheritance(
        childId,
        newChildOutputChannels as number,
        updatedNodes,
        currentEdges,
        defaultLayers,
        defaultTensorOps,
        defaultActivators,
        defaultInputs
      );
    }
  });
  
  return updatedNodes;
};

// Function to handle inherit_from_parent parameter changes
export const handleInheritFromParentChange = (
  nodeId: string,
  setEdges: (callback: (edges: any[]) => any[]) => void,
  setNodes: (callback: (nodes: any[]) => any[]) => void,
  defaultLayers: any,
  defaultTensorOps: any,
  defaultActivators: any,
  defaultInputs: any
) => {
  // Use setTimeout to access current edges state
  setTimeout(() => {
    setEdges((currentEdges) => {
      setNodes((currentNodes) => {
        // Find parent node via incoming edges
        const parentEdge = currentEdges.find(edge => edge.target === nodeId);
        
        if (parentEdge) {
          const parentNode = currentNodes.find(node => node.id === parentEdge.source);
          if (parentNode && parentNode.data.outputChannels !== undefined) {
            // Update the specific node with inherited values
            const updatedNodes = currentNodes.map(node => {
              if (node.id === nodeId) {
                const nodeDefinition = findNodeDefinition(
                  node.data.operationType, 
                  node.data.type,
                  defaultLayers,
                  defaultTensorOps,
                  defaultActivators,
                  defaultInputs
                );
                
                let inheritedData = { ...node.data };
                
                // Update input channels to match parent's output
                inheritedData.inputChannels = parentNode.data.outputChannels;
                
                // Update linked input parameter
                const channelLinks = nodeDefinition?.parseCheck?.ChannelLinks || [];
                channelLinks.forEach((link: any) => {
                  if (link.inputParam) {
                    inheritedData.parameters = {
                      ...inheritedData.parameters,
                      [link.inputParam]: parentNode.data.outputChannels
                    };
                  }
                });
                
                return { ...node, data: inheritedData };
              }
              return node;
            });
            
            return updatedNodes;
          }
        }
        return currentNodes;
      });
      return currentEdges; // Don't modify edges
    });
  }, 0);
};
