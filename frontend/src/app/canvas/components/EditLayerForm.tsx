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

    const deleteNodeLocal = () => {
        deleteNode(selectedNodes[0].id)
    }

    return (
        selectedNodes[0] != null && (
            <Box sx={{ p: 2 }}>
                {/* Title for the form */}
                <Typography variant="subtitle1" sx={{ mb: 2 }}>
                    Add Layer
                </Typography>
                {/* Input for the layer label */}
                <TextField
                    label="Layer label"
                    value={selectedNodes[0].data.label}
                    onChange={(e) => {updateNodeLabel(selectedNodes[0].id, e.target.value)}}
                    fullWidth
                    size="small"
                    sx={{ mb: 2 }}
                />
                <TextField
                    select
                    label="Operation Type"
                    value={selectedNodes[0].data.operationType}
                    onChange={(e) => {updateNodeOperationType(selectedNodes[0].id, e.target.value)}}
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
                        selectedNodes[0].data.operationType === "Layer" ? "Layer Type" : 
                        selectedNodes[0].data.operationType === "TensorOp" ? "Tensor Operation Type" 
                        : "Activator Type"
                     }
                    value={selectedNodes[0].data.type}
                    onChange={(e) => {updateNodeType(selectedNodes[0].id, selectedNodes[0].data.operationType, e.target.value)}}
                    fullWidth
                    size="small"
                    sx={{ mb: 2 }}
                >   
                    {(selectedNodes[0].data.operationType === "Layer") && 
                        defaultLayers.map((dLayer) => (
                            <MenuItem key={dLayer.type} value={dLayer.type}>{dLayer.type}</MenuItem>
                    ))}
                    {(selectedNodes[0].data.operationType === "TensorOp") && 
                        defaultTensorOps.map((dLayer) => (
                            <MenuItem key={dLayer.type} value={dLayer.type}>{dLayer.type}</MenuItem>
                    ))}
                    {(selectedNodes[0].data.operationType === "Activator") && 
                        defaultActivators.map((dLayer) => (
                            <MenuItem key={dLayer.type} value={dLayer.type}>{dLayer.type}</MenuItem>
                    ))}
                </TextField>
                
                {   Object.keys(selectedNodes[0].data.parameters).map((parameterKey, i) => (
                        <TextField
                            key={i}
                            label={parameterKey}
                            value={selectedNodes[0].data.parameters[parameterKey]}
                            onChange={(e) => {updateNodeParameter(selectedNodes[0].id, parameterKey, e.target.value)}}
                            type={typeof selectedNodes[0].data.parameters[parameterKey] === "number" ? "number" : "text"}
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
        )
    );
}

