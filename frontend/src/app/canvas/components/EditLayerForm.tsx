"use client";

import { useState, useEffect } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import MenuItem from "@mui/material/MenuItem";
import Button from "@mui/material/Button";
import ParameterInputs from "./ParameterInputs";

// Define the props expected by the LayerForm component
interface Props {
    // shows the selected nodes at a given time
    selectedNodes: any[];
    // the defaultnodes available,
    defaultLayers: any[];
    defaultActivators: any[];
    defaultTensorOps: any[];
    // allows the update of layerType
    updateNodeType: (targetID: any, valA: any, valB: any) => void;
    // allows for the update of operationType
    updateNodeOperationType: (targetID: any, val: any) => void;
    // allows for the update of node paramters
    updateNodeParameter: (targetID: any, valA: any, valB: any) => void;
    // allows for the deletion of the selectedNode
    deleteNode: (targetID: any) => void;
}

// The LayerForm component allows users to add a new layer to the canvas
export default function EditLayerForm({selectedNodes, defaultActivators, defaultTensorOps, defaultLayers, updateNodeType, updateNodeOperationType, updateNodeParameter , deleteNode}: Props) {
    // State for validation errors
    const [hasValidationErrors, setHasValidationErrors] = useState(false);

    // Early return if no nodes are selected
    if (!selectedNodes || selectedNodes.length === 0 || !selectedNodes[0]) {
        return null;
    }

    const selectedNode = selectedNodes[0];

    const deleteNodeLocal = () => {
        deleteNode(selectedNode.id);
    };

    // Handle parameter changes from ParameterInputs component
    const handleParameterChange = (parameterKey: string, value: any) => {
        updateNodeParameter(selectedNode.id, parameterKey, value);
    };

    // Handle validation state changes from ParameterInputs component
    const handleValidationChange = (hasErrors: boolean) => {
        setHasValidationErrors(hasErrors);
    };

    return (
        <Box sx={{ p: 2 }}>
            {/* Title for the form */}
            <Typography variant="subtitle1" sx={{ mb: 2 }}>
                Edit Node
            </Typography>
                <TextField
                    select
                    label="Operation Type"
                    value={selectedNode.data.operationType || ""}
                    onChange={(e) => {updateNodeOperationType(selectedNode.id, e.target.value)}}
                    fullWidth
                    size="small"
                    sx={{ mb: 2 }}
                >   
                        <MenuItem key={"Layer"} value={"Layer"}>{"Layer"}</MenuItem>
                        <MenuItem key={"TensorOp"} value={"TensorOp"}>{"Tensor Operation"}</MenuItem>
                        <MenuItem key={"Activator"} value={"Activator"}>{"Activator"}</MenuItem>
                </TextField>

                <TextField
                    select
                    label={
                        selectedNode.data.operationType === "Layer" ? "Layer Type" : 
                        selectedNode.data.operationType === "TensorOp" ? "Tensor Operation Type" 
                        : "Activator Type"
                     }
                    value={selectedNode.data.type || ""}
                    onChange={(e) => {updateNodeType(selectedNode.id, selectedNode.data.operationType, e.target.value)}}
                    fullWidth
                    size="small"
                    sx={{ mb: 2 }}
                >   
                    {(selectedNode.data.operationType === "Layer") && 
                        defaultLayers.map((dLayer) => (
                            <MenuItem key={dLayer.type} value={dLayer.type}>{dLayer.type}</MenuItem>
                    ))}
                    {(selectedNode.data.operationType === "TensorOp") && 
                        defaultTensorOps.map((dLayer) => (
                            <MenuItem key={dLayer.type} value={dLayer.type}>{dLayer.type}</MenuItem>
                    ))}
                    {(selectedNode.data.operationType === "Activator") && 
                        defaultActivators.map((dLayer) => (
                            <MenuItem key={dLayer.type} value={dLayer.type}>{dLayer.type}</MenuItem>
                    ))}
                </TextField>
                
                {selectedNode.data.parameters && (
                    <ParameterInputs
                        operationType={selectedNode.data.operationType}
                        nodeType={selectedNode.data.type}
                        parameters={selectedNode.data.parameters}
                        onParameterChange={handleParameterChange}
                        onValidationChange={handleValidationChange}
                    />
                )}
                
                <Button variant="contained" fullWidth style={{backgroundColor: "red"}} onClick={deleteNodeLocal}>
                    Delete
                </Button>
            </Box>
    );
}

