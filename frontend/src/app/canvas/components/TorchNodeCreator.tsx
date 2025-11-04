import {Node} from "@xyflow/react";

export const createNode = (
    id, 
    posModifier, 
    label, 
    operationType,
    type, 
    parameters, 
    getSetters,
    getDefaults
): Node => {
    return {
        id: id,
        position: { x: 100, y: 100 + posModifier * 60 },
        type: "torchNode",
        data: {
            errors: [], // So the node can display its errors (TN)
            getSetters: getSetters,
            getDefaults: getDefaults,
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
