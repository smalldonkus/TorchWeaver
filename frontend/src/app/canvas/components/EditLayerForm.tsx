"use client";

import { useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import MenuItem from "@mui/material/MenuItem";
import Button from "@mui/material/Button";

// Define the props expected by the LayerForm component
interface Props {
    // nodes: an array representing the current layers/nodes in the canvas
    nodes: any[];
    // setNodes: a function to update the nodes array in the parent component
    setNodes: (val: any) => void;
    // shows the selected nodes at a given time
    selectedNodes: any[];
    // allows the update of Label
    updateNodeLabel: (targetID: any, val: any) => void;
    // allows the update of layerType
    updateNodeLayerType: (targetID: any, val: any) => void;
}

// The LayerForm component allows users to add a new layer to the canvas
export default function EditLayerForm({ nodes, setNodes, selectedNodes, updateNodeLabel, updateNodeLayerType }: Props) {

    // if (selectedNodes[0] != null) {console.log(selectedNodes[0].data.label)}
    // Render the form UI for adding a new layer
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
                {/* Dropdown to select the layer type */}
                <TextField
                    select
                    label="Layer Type"
                    value={selectedNodes[0].data.layerType}
                    onChange={(e) => {updateNodeLayerType(selectedNodes[0].id, e.target.value)}}
                    fullWidth
                    size="small"
                    sx={{ mb: 2 }}
                >
                    <MenuItem value="Linear">Linear</MenuItem>
                    <MenuItem value="Convolutional">Convolutional</MenuItem>
                    <MenuItem value="Flatten">Flatten</MenuItem>
                </TextField>

                {/** TODO: make custom param setters */}
            </Box>
        )
    );
}

