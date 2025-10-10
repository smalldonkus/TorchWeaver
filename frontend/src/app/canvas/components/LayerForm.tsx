"use client";

// Import React hooks and MUI components
import { useState, useEffect } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import MenuItem from "@mui/material/MenuItem";
import Button from "@mui/material/Button";

// Define the props that this component expects
interface Props {
    nodes: any[]; // List of current nodes (layers)
    setNodes: (val: any) => void; // Function to update nodes
    defaultLayers: any[]; // List of available layer types
}

// Main component for the Layer Form
export default function LayerForm({ nodes, setNodes, defaultLayers }: Props) {
    // If layer types haven't loaded yet, show a loading message
    if (!defaultLayers || defaultLayers.length === 0) {
        return <div>Loading layer types...</div>;
    }

    // State for the new layer's label (name)
    const [newLabel, setNewLabel] = useState("");
    // State for the currently selected default layer type
    const [chosenDefault, setChosenDefault] = useState(defaultLayers[0]);

    // When defaultLayers changes (e.g., after loading), update chosenDefault
    useEffect(() => {
        setChosenDefault(defaultLayers[0]);
    }, [defaultLayers]);

    // Change the selected layer type when user picks a new one
    function setLayer(layerName: string) {
        setChosenDefault(defaultLayers.find((layer) => layer.type === layerName));
    }

    // Update a parameter value for the selected layer type
    const updateParam = (paramaterKey: string, parameterValue: string) => {
        setChosenDefault(prev => ({
            ...prev,
            parameters: { ...prev.parameters, [paramaterKey]: parameterValue }
        }));
    };

    // Add a new layer node to the list
    const addLayer = () => {
        const newId = `n${nodes.length + 1}`; // Generate a new id
        setNodes([
            ...nodes,
            {
                id: newId,
                position: { x: 100, y: 100 + nodes.length * 60 }, // Position it below previous nodes
                data: {
                    label: `${chosenDefault.layerType || chosenDefault.type}: ${newLabel || `Node ${nodes.length + 1}`}`,
                    operationType: "Layer",
                    layerType: chosenDefault.type,
                    parameters: chosenDefault.parameters
                },
            },
        ]);
        setNewLabel(""); // Reset label input
        setChosenDefault(defaultLayers[0]); // Reset to first layer type
    };

    // Render the form UI
    return (
        <Box sx={{ p: 2 }}>
            {/* Title */}
            <Typography variant="subtitle1" sx={{ mb: 2 }}>
                Add Layer
            </Typography>
            {/* Input for layer label */}
            <TextField
                label="Layer label"
                value={newLabel}
                onChange={(e) => setNewLabel(e.target.value)}
                fullWidth
                size="small"
                sx={{ mb: 2 }}
            />
            {/* Dropdown to select layer type */}
            <TextField
                select
                label="Layer Type"
                value={chosenDefault?.type}
                onChange={(e) => setLayer(e.target.value)}
                fullWidth
                size="small"
                sx={{ mb: 2 }}
            >
                {/* Show all available layer types */}
                {defaultLayers.map((dLayer) => (
                    <MenuItem key={dLayer.type} value={dLayer.type}>{dLayer.type}</MenuItem>
                ))}
            </TextField>
            {/* Show parameter inputs for the selected layer type */}
            {chosenDefault && chosenDefault.parameters &&
                Object.keys(chosenDefault.parameters).map((paramterKey, i) => (
                    <TextField
                        key={i}
                        label={paramterKey}
                        value={chosenDefault.parameters[paramterKey]}
                        onChange={(e) => updateParam(paramterKey, e.target.value)}
                        type={typeof chosenDefault.parameters[paramterKey] === "number" ? "number" : "text"}
                        fullWidth
                        size="small"
                        sx={{ mb: 2 }}
                    />
                ))
            }
            {/* Button to add the new layer */}
            <Button variant="contained" fullWidth onClick={addLayer}>
                Add Layer
            </Button>
        </Box>
    );
}