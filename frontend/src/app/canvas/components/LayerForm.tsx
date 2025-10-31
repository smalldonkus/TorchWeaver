"use client";

// Import React hooks and MUI components
import { useState, useEffect } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import MenuItem from "@mui/material/MenuItem";
import Button from "@mui/material/Button";
import { generateUniqueNodeId } from "../utils/idGenerator";
import ParameterInputs from "./ParameterInputs";
import { useNodeDefinitions } from "../hooks/useNodeDefinitions";
import { useVariablesInfo } from "../hooks/useVariablesInfo";

// Define the props that this component expects
interface Props {
    nodes: any[]; // List of current nodes (layers)
    setNodes: (val: any) => void; // Function to update nodes
    defaultLayers: any[]; // List of available layer types
}

// Main component for the Layer Form
export default function LayerForm({ nodes, setNodes, defaultLayers }: Props) {
    // All hooks must be called before any conditional returns!
    
    // Fetch layer definitions and variables info from backend
    const { loading: layerDefsLoading, error: layerDefsError } = useNodeDefinitions("layers");
    const { loading: variablesLoading, error: variablesError } = useVariablesInfo();

    // State for the currently selected default layer type
    const [chosenDefault, setChosenDefault] = useState(defaultLayers?.[0] || null);
    
    // State for validation errors
    const [hasValidationErrors, setHasValidationErrors] = useState(false);

    // When defaultLayers changes (e.g., after loading), update chosenDefault
    useEffect(() => {
        if (defaultLayers && defaultLayers.length > 0) {
            setChosenDefault(defaultLayers[0]);
        }
    }, [defaultLayers]);

    // Now handle conditional rendering AFTER all hooks
    
    // Show error if API calls failed
    if (layerDefsError || variablesError) {
        return (
            <div>
                <Typography color="error">
                    Error loading data: {layerDefsError || variablesError}
                </Typography>
            </div>
        );
    }
    
    // If layer types haven't loaded yet, show a loading message
    if (!defaultLayers || defaultLayers.length === 0 || layerDefsLoading || variablesLoading) {
        return <div>Loading layer types...</div>;
    }

    // Change the selected layer type when user picks a new one
    function setLayer(layerName: string) {
        const newLayer = defaultLayers.find((layer) => layer.type === layerName);
        setChosenDefault(newLayer);
    }

    // Handle parameter changes from ParameterInputs component
    const handleParameterChange = (parameterKey: string, value: any) => {
        setChosenDefault(prev => prev ? ({
            ...prev,
            parameters: { 
                ...prev.parameters, 
                [parameterKey]: value 
            }
        }) : null);
    };

    // Handle validation state changes from ParameterInputs component
    const handleValidationChange = (hasErrors: boolean) => {
        setHasValidationErrors(hasErrors);
    };

    // Add a new layer node to the list
    const addLayer = () => {
        // Check if there are any validation errors
        if (hasValidationErrors) {
            alert("Please fix parameter errors before adding the layer.");
            return;
        }

        const newId = generateUniqueNodeId("layer", nodes);
        setNodes([
            ...nodes,
            {
                id: newId,
                position: { x: 100, y: 100 + nodes.length * 60 },
                data: {
                    label: chosenDefault.type,
                    operationType: "Layer",
                    type: chosenDefault.type,
                    parameters: chosenDefault.parameters,
                    outgoing_edges_count: 0
                },
            },
        ]);
        setChosenDefault(defaultLayers[0]);
    };

    // Render the form UI
    return (
        <Box sx={{ p: 2 }}>
            {/* Title */}
            <Typography variant="subtitle1" sx={{ mb: 2 }}>
                Add Layer
            </Typography>
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
            {chosenDefault && (
                <ParameterInputs
                    operationType="Layer"
                    nodeType={chosenDefault.type}
                    parameters={chosenDefault.parameters}
                    onParameterChange={handleParameterChange}
                    onValidationChange={handleValidationChange}
                />
            )}
            
            {/* Button to add the new layer */}
            <Button variant="contained" fullWidth onClick={addLayer}>
                Add Layer
            </Button>
        </Box>
    );
}