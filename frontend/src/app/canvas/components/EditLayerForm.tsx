"use client";

import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import MenuItem from "@mui/material/MenuItem";
import Button from "@mui/material/Button";

// Define the props expected by the LayerForm component
interface Props {
    // shows the selected nodes at a given time
    selectedNodes: any[];
    // the defaultnodes available,
    defaultLayers: any[];
    defaultActivators: any[];
    defaultTensorOps: any[];
    // allows the update of Label
    updateNodeLabel: (targetID: any, val: any) => void;
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
export default function EditLayerForm({selectedNodes, defaultActivators, defaultTensorOps, defaultLayers, updateNodeLabel, updateNodeType, updateNodeOperationType, updateNodeParameter , deleteNode}: Props) {

    // Early return if no nodes are selected
    if (!selectedNodes || selectedNodes.length === 0 || !selectedNodes[0]) {
        return null;
    }

    const selectedNode = selectedNodes[0];

    const deleteNodeLocal = () => {
        deleteNode(selectedNode.id);
    };

    return (
        <Box sx={{ p: 2 }}>
            {/* Title for the form */}
            <Typography variant="subtitle1" sx={{ mb: 2 }}>
                Edit Node
            </Typography>
            {/* Input for the layer label */}
            <TextField
                label="Layer label"
                value={selectedNode.data.label || ""}
                onChange={(e) => {updateNodeLabel(selectedNode.id, e.target.value)}}
                fullWidth
                size="small"
                sx={{ mb: 2 }}
            />
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
                
                {selectedNode.data.parameters && Object.keys(selectedNode.data.parameters).map((parameterKey, i) => (
                        <TextField
                            key={i}
                            label={parameterKey}
                            value={selectedNode.data.parameters[parameterKey] || ""}
                            onChange={(e) => {updateNodeParameter(selectedNode.id, parameterKey, e.target.value)}}
                            type={typeof selectedNode.data.parameters[parameterKey] === "number" ? "number" : "text"}
                            fullWidth
                            size="small"
                            sx={{ mb: 2 }}
                        />
                    ))
                }
                <Button variant="contained" fullWidth style={{backgroundColor: "red"}} onClick={deleteNodeLocal}>
                    Delete
                </Button>
            </Box>
    );
}

