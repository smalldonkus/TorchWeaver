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
}

// The LayerForm component allows users to add a new layer to the canvas
export default function LayerForm({ nodes, setNodes }: Props) {
    // State for the label of the new layer
    const [newLabel, setNewLabel] = useState("");
    // State for the type of layer (e.g., Linear, Convolutional, Flatten)
    const [layerType, setLayerType] = useState("Linear");
    // State for the number of input features (used for Linear layers)
    const [inFeatures, setInFeatures] = useState("");
    // State for the number of output features (used for Linear layers)
    const [outFeatures, setOutFeatures] = useState("");

    // Function to add a new layer to the nodes array
    const addLayer = () => {
        // Generate a unique id for the new node based on the current number of nodes
        const newId = `n${nodes.length + 1}`;
        // Update the nodes array by adding the new node with its properties
        setNodes([
            ...nodes,
            {
                id: newId,
                // Position the new node below the previous ones
                position: { x: 100, y: 100 + nodes.length * 60 },
                data: {
                    // Set the label to the selected layer type and user-provided label, or a default if empty
                    label: `${layerType}: ${newLabel || `Node ${nodes.length + 1}`}`,
                    // Indicate this node is a Layer operation
                    operationType: "Layer",
                    // Store the selected layer type
                    layerType,
                    // Store parameters relevant to the layer (only used for Linear here)
                    parameters: {
                        in_features: inFeatures,
                        out_features: outFeatures,
                    },
                },
            },
        ]);
        // Reset the form fields after adding the layer
        setNewLabel("");
        setInFeatures("");
        setOutFeatures("");
    };

    // Render the form UI for adding a new layer
    return (
        <Box sx={{ p: 2 }}>
            {/* Title for the form */}
            <Typography variant="subtitle1" sx={{ mb: 2 }}>
                Add Layer
            </Typography>
            {/* Input for the layer label */}
            <TextField
                label="Layer label"
                value={newLabel}
                onChange={(e) => setNewLabel(e.target.value)}
                fullWidth
                size="small"
                sx={{ mb: 2 }}
            />
            {/* Dropdown to select the layer type */}
            <TextField
                select
                label="Layer Type"
                value={layerType}
                onChange={(e) => setLayerType(e.target.value)}
                fullWidth
                size="small"
                sx={{ mb: 2 }}
            >
                <MenuItem value="Linear">Linear</MenuItem>
                <MenuItem value="Convolutional">Convolutional</MenuItem>
                <MenuItem value="Flatten">Flatten</MenuItem>
            </TextField>
            {/* Show extra fields for Linear layers only */}
            {layerType === "Linear" && (
                <>
                    {/* Input for number of input features */}
                    <TextField
                        label="In Features"
                        value={inFeatures}
                        onChange={(e) => setInFeatures(e.target.value)}
                        type="number"
                        fullWidth
                        size="small"
                        sx={{ mb: 2 }}
                    />
                    {/* Input for number of output features */}
                    <TextField
                        label="Out Features"
                        value={outFeatures}
                        onChange={(e) => setOutFeatures(e.target.value)}
                        type="number"
                        fullWidth
                        size="small"
                        sx={{ mb: 2 }}
                    />
                </>
            )}
            {/* Button to add the new layer */}
            <Button variant="contained" fullWidth onClick={addLayer}>
                Add Layer
            </Button>
        </Box>
    );
}



