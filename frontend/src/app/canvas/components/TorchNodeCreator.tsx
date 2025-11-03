import {Node} from "@xyflow/react";

export const createNode = (
    id, 
    posModifier, 
    label, 
    operationType,
    type, 
    parameters, 
    updateNodeParameter, 
    updateNodeType, 
    updateNodeOperationType, 
    deleteNode,
    getDefaults
): Node => {
    return {
        id: id,
        position: { x: 100, y: 100 + posModifier * 60 },
        type: "torchNode",
        data: {
            errors: [], // So the node can display its errors (TN)
            setters: { 
            // the node needs a reference to these, 
            // since nothing can be given to the node once its created
            // for functionality like editing, deleting itself (TN)
                updateNodeParameter: updateNodeParameter,
                updateNodeType: updateNodeType,
                updateNodeOperationType: updateNodeOperationType,
                deleteNode: deleteNode,
                getDefaults: getDefaults // doesn't fit the category "setters", but this is the best place for it.
            },
            label: label,
            operationType: operationType,
            type: type,
            parameters: parameters,
            outgoing_edges_count: 0
        },
        style : {
            borderRadius: "5px",
            border: "1px solid black",
            padding: "5px",
            backgroundColor: "white"
        }
    }
}
