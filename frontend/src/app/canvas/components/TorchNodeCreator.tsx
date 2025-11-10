import {Node} from "@xyflow/react";

export const determineCanInheritFromParent = (
  nodeDefinition: any,
  parameters: Record<string, any>
): boolean => {
  const canEditChannels = nodeDefinition?.parseCheck?.CanEditChannels;
  
  if (canEditChannels) {
    // If node can edit channels, only inherit if user explicitly sets inherit_from_parent to true
    return parameters.inherit_from_parent || false;
  } else {
    // If node can't edit channels, always inherit (automatic inheritance)
    return true;
  }
};

/**
 * Links parameters to channel values based on ChannelLinks definition
 * @param nodeDefinition - The node definition containing ChannelLinks
 * @param parameters - The node's parameters
 * @returns Object with inputChannels and outputChannels values
 */
export const linkParametersToChannels = (
  nodeDefinition: any,
  parameters: Record<string, any>
): { inputChannels: number; outputChannels: number } => {
  let inputChannels = 1;  // default value
  let outputChannels = 1; // default value
  
  // If CanEditChannels is true, link parameters to channels
  if (nodeDefinition?.parseCheck?.CanEditChannels) {
    nodeDefinition.parseCheck.ChannelLinks?.forEach((link: any) => {
      if (link.inputParam && parameters[link.inputParam] !== undefined) {
        inputChannels = parameters[link.inputParam];
      }
      if (link.outputParam && parameters[link.outputParam] !== undefined) {
        outputChannels = parameters[link.outputParam];
      }
    });
  }

  return { inputChannels, outputChannels };
};

export const createNode = (
    id, 
    posModifier, 
    label, 
    operationType,
    type, 
    parameters, 
    getSetters,
    getDefaults,
    chosenDefault = null
): Node => {
    // Calculate hidden attributes for non-output nodes
    const isOutputNode = operationType === "Output";
    
    let inputChannels: number | null = null;
    let outputChannels: number | null = null;
    let canInheritFromParent = false;

    if (!isOutputNode && chosenDefault) {
        const channelData = linkParametersToChannels(chosenDefault, parameters);
        inputChannels = channelData.inputChannels;
        outputChannels = channelData.outputChannels;
        canInheritFromParent = determineCanInheritFromParent(chosenDefault, parameters);
    }

    const nodeData: any = {
        errors: [], // So the node can display its errors (TN)
        getSetters: getSetters,
        getDefaults: getDefaults,
        label: label,
        operationType: operationType,
        type: type,
        parameters: parameters,
        outgoing_edges_count: 0,
    };

    // Add hidden attributes for non-output nodes
    if (!isOutputNode) {
        nodeData.inputChannels = inputChannels;
        nodeData.outputChannels = outputChannels;
        nodeData.can_inherit_from_parent = canInheritFromParent;
    }

    return {
        id: id,
        position: { x: 100, y: 100 + posModifier * 60 },
        type: "torchNode",
        data: nodeData,
        style : {
            borderRadius: "5px",
            border: "1px solid black",
            padding: "5px",
            backgroundColor: "white"
        }
    }
}
