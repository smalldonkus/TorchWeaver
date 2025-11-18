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
import { useParameterHandling } from "../hooks/useParameterHandling";

import {createNode} from "./TorchNodeCreator";


// Define the props that this component expects
interface Props {
    nodes: any[]; // List of current nodes (layers)
    addNode: (val: any) => void; // Function to update nodes
    defaultLayers: any; // Layer data with global classes structure

    // for TorchNode functionality, allows it to update itself (TN)
    getSetters: () => any;
    getDefaults: () => any; // for editing within a node (TN)
}

// Main component for the Layer Form
export default function LayerForm({ nodes, addNode, defaultLayers, getSetters, getDefaults}: Props) {
    // All hooks must be called before any conditional returns!
    
    // Use parameter handling hook
    const { 
        parameters, 
        hasValidationErrors, 
        handleParameterChange, 
        handleValidationChange, 
        updateParameters 
    } = useParameterHandling();
    
    // State for the currently selected global class and layer type
    const [selectedClass, setSelectedClass] = useState<string>("");
    const [selectedLayerType, setSelectedLayerType] = useState<string>("");
    const [chosenDefault, setChosenDefault] = useState<any>(null);

    // Extract global classes from the new structure
    const globalClasses = defaultLayers?.data ? Object.keys(defaultLayers.data) : [];

    // When defaultLayers changes, set initial selections
    useEffect(() => {
        if (globalClasses.length > 0) {
            setSelectedClass(globalClasses[0]);
            const firstClassLayers = defaultLayers.data[globalClasses[0]];
            const firstLayerType = Object.keys(firstClassLayers)[0];
            setSelectedLayerType(firstLayerType);
            setChosenDefault({
                class: globalClasses[0],
                type: firstLayerType,
                ...firstClassLayers[firstLayerType]
            });
            updateParameters(firstClassLayers[firstLayerType]?.parameters || {});
        }
    }, [defaultLayers, updateParameters]);

    // If layer types haven't loaded yet, show a loading message
    if (!defaultLayers || !defaultLayers.data || Object.keys(defaultLayers.data).length === 0) {
        return <div>Loading layer types...</div>;
    }

    // Handle class selection change
    function handleClassChange(className: string) {
        setSelectedClass(className);
        const classLayers = defaultLayers.data[className];
        const firstLayerType = Object.keys(classLayers)[0];
        setSelectedLayerType(firstLayerType);
        setChosenDefault({
            class: className,
            type: firstLayerType,
            ...classLayers[firstLayerType]
        });
        updateParameters(classLayers[firstLayerType]?.parameters || {});
    }

    // Handle layer type selection change within a class
    function handleLayerTypeChange(layerType: string) {
        setSelectedLayerType(layerType);
        const layerData = defaultLayers.data[selectedClass][layerType];
        setChosenDefault({
            class: selectedClass,
            type: layerType,
            ...layerData
        });
        updateParameters(layerData?.parameters || {});
    }

    // Add a new layer node to the list
    const addLayer = () => {
        // Check if there are any validation errors
        if (hasValidationErrors) {
            alert("Please fix parameter errors before adding the layer.");
            return;
        }

        const newId = generateUniqueNodeId("layer", nodes);
        const newNode = createNode(
            newId,
            nodes.length, // posModifier
            chosenDefault.type, // label
            "Layer", // operation type
            chosenDefault.type, // type
            parameters,
            getSetters,
            getDefaults,
            chosenDefault
        );
        addNode([
            ...nodes,
            newNode
        ]);
        
        // Reset to first selection after adding
        if (globalClasses.length > 0) {
            setSelectedClass(globalClasses[0]);
            const firstClassLayers = defaultLayers.data[globalClasses[0]];
            const firstLayerType = Object.keys(firstClassLayers)[0];
            setSelectedLayerType(firstLayerType);
            setChosenDefault({
                class: globalClasses[0],
                type: firstLayerType,
                ...firstClassLayers[firstLayerType]
            });
            updateParameters(firstClassLayers[firstLayerType]?.parameters || {});
        }
    };

    // Render the form UI
    return (
        <Box sx={{ p: 2 }}>
            {/* Title */}
            <Typography variant="subtitle1" sx={{ mb: 2 }}>
                Add Layer
            </Typography>
            
            {/* Dropdown to select layer class */}
            <TextField
                select
                label="Layer Class"
                value={selectedClass}
                onChange={(e) => handleClassChange(e.target.value)}
                fullWidth
                size="small"
                sx={{ mb: 2 }}
            >
                {globalClasses.map((className) => (
                    <MenuItem key={className} value={className}>{className}</MenuItem>
                ))}
            </TextField>

            {/* Dropdown to select specific layer type within the class */}
            {selectedClass && (
                <TextField
                    select
                    label="Layer Type"
                    value={selectedLayerType}
                    onChange={(e) => handleLayerTypeChange(e.target.value)}
                    fullWidth
                    size="small"
                    sx={{ mb: 2 }}
                >
                    {Object.keys(defaultLayers.data[selectedClass]).map((layerType) => (
                        <MenuItem key={layerType} value={layerType}>{layerType}</MenuItem>
                    ))}
                </TextField>
            )}
            
            {/* Show parameter inputs for the selected layer type */}
            {chosenDefault && (
                <ParameterInputs
                    operationType="Layer"
                    nodeType={chosenDefault.type}
                    parameters={parameters}
                    onParameterChange={handleParameterChange}
                    onValidationChange={handleValidationChange}
                    nodeDefinition={chosenDefault}
                />
            )}
            
            {/* Button to add the new layer */}
            <Button 
                variant="contained" 
                fullWidth 
                onClick={addLayer}
                sx={{ 
                    backgroundColor: '#202A44',
                    borderRadius: '8px',
                    textTransform: 'none',
                    fontWeight: 600,
                    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                    '&:hover': {
                        backgroundColor: '#2d3a5e',
                        boxShadow: '0 4px 8px rgba(0,0,0,0.2)'
                    }
                }}
            >
                Add Layer
            </Button>
        </Box>
    );
}