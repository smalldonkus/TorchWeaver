"use client";

import { useState, useEffect } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import MenuItem from "@mui/material/MenuItem";
import Button from "@mui/material/Button";
import ParameterInputs from "./ParameterInputs";
import { useParameterHandling } from "../hooks/useParameterHandling";

// Define the props expected by the LayerForm component
interface Props {
    // shows the selected nodes at a given time
    selectedNodes: any[];
    // the defaultnodes available,
    defaultLayers: any; // Changed from any[] to any for new structure
    defaultActivators: any; // Changed from any[] to any for new structure
    defaultTensorOps: any; // Changed from any[] to any for new structure
    defaultInputs: any; // Input definitions data structure
    // allows the update of layerType
    updateNodeType: (targetID: any, valA: any, valB: any, valC: any) => void;
    // allows for the update of operationType
    updateNodeOperationType: (targetID: any, valA: any, valB: any,  valC: any) => void;
    // allows for the update of node paramters
    updateNodeParameter: (targetID: any, valA: any, valB: any) => void;
    // allows for the deletion of the selectedNode
    deleteNode: (targetID: any) => void;
}

// The LayerForm component allows users to add a new layer to the canvas
export default function EditLayerForm({selectedNodes, defaultActivators, defaultTensorOps, defaultLayers, defaultInputs, updateNodeType, updateNodeOperationType, updateNodeParameter , deleteNode}: Props) {
    
    // Use parameter handling hook
    const { 
        parameters, 
        hasValidationErrors, 
        handleParameterChange, 
        handleValidationChange, 
        updateParameters 
    } = useParameterHandling();
    
    // State for hierarchical selection
    const [selectedOperationType, setSelectedOperationType] = useState<string>("");
    const [selectedClass, setSelectedClass] = useState<string>("");
    const [selectedSpecificType, setSelectedSpecificType] = useState<string>("");
    
    // State for tracking pending changes
    const [hasPendingChanges, setHasPendingChanges] = useState(false);

    // Get the selected node (but after hooks are declared)
    const selectedNode = selectedNodes && selectedNodes.length > 0 ? selectedNodes[0] : null;

    // Helper functions for getting available options
    const getAvailableClasses = (operationType: string): string[] => {
        let data;
        switch (operationType) {
            case "Layer":
                data = defaultLayers;
                break;
            case "TensorOp":
                data = defaultTensorOps;
                break;
            case "Activator":
                data = defaultActivators;
                break;
            case "Input":
                data = defaultInputs;
                break;
            default:
                return [];
        }
        return data?.data ? Object.keys(data.data) : [];
    };

    // Get available specific types based on operation type and class
    const getAvailableSpecificTypes = (operationType: string, className: string): string[] => {
        let data;
        switch (operationType) {
            case "Layer":
                data = defaultLayers;
                break;
            case "TensorOp":
                data = defaultTensorOps;
                break;
            case "Activator":
                data = defaultActivators;
                break;
            case "Input":
                data = defaultInputs;
                break;
            default:
                return [];
        }
        return data?.data?.[className] ? Object.keys(data.data[className]) : [];
    };

    // Initialize state when selectedNode changes
    useEffect(() => {
        if (selectedNode) {
            setSelectedOperationType(selectedNode.data.operationType || "");
            setSelectedSpecificType(selectedNode.data.type || "");
            updateParameters(selectedNode.data.parameters || {});
            setHasPendingChanges(false);
        }
    }, [selectedNode, updateParameters, selectedNode?.data.operationType, selectedNode?.data.type, selectedNode?.data.parameters]);

    // Initialize selected class based on current type
    useEffect(() => {
        if (selectedOperationType && selectedSpecificType) {
            // Find which class contains the current specific type
            const availableClasses = getAvailableClasses(selectedOperationType);
            for (const className of availableClasses) {
                const specificTypes = getAvailableSpecificTypes(selectedOperationType, className);
                if (specificTypes.includes(selectedSpecificType)) {
                    setSelectedClass(className);
                    break;
                }
            }
        }
    }, [selectedOperationType, selectedSpecificType, defaultLayers, defaultTensorOps, defaultActivators, defaultInputs]);

    // Early return AFTER all hooks are declared
    if (!selectedNode) {
        return null;
    }

    const deleteNodeLocal = () => {
        deleteNode(selectedNode.id);
    };

    // Handle operation type change
    const handleOperationTypeChange = (newOperationType: string) => {
        setSelectedOperationType(newOperationType);
        setSelectedClass("");
        setSelectedSpecificType("");
        setHasPendingChanges(true);
    };

    // Handle class change
    const handleClassChange = (newClass: string) => {
        setSelectedClass(newClass);
        setSelectedSpecificType("");
        setHasPendingChanges(true);
    };

    // Handle specific type change
    const handleSpecificTypeChange = (newSpecificType: string) => {
        setSelectedSpecificType(newSpecificType);
        
        // Load default parameters for the new type
        let dataSource;
        switch (selectedOperationType) {
            case "Layer":
                dataSource = defaultLayers;
                break;
            case "TensorOp":
                dataSource = defaultTensorOps;
                break;
            case "Activator":
                dataSource = defaultActivators;
                break;
            case "Input":
                dataSource = defaultInputs;
                break;
            default:
                dataSource = null;
        }
        
        const newTypeDefinition = dataSource?.data?.[selectedClass]?.[newSpecificType];
        if (newTypeDefinition?.parameters) {
            // The parameters object already contains the default values
            updateParameters({ ...newTypeDefinition.parameters });
        } else {
            updateParameters({});
        }
        
        setHasPendingChanges(true);
    };

    // Wrap the parameter change handler to track pending changes
    const handleParameterChangeWithPending = (parameterKey: string, value: any) => {
        handleParameterChange(parameterKey, value);
        setHasPendingChanges(true);
    };

    // Apply all pending changes
    const handleApplyEdit = () => {
        if (!selectedNode) return;
       
        if ((selectedOperationType && selectedOperationType !== selectedNode.data.operationType)
            &&
            (selectedSpecificType  && selectedSpecificType  !== selectedNode.data.type)){
            updateNodeOperationType(selectedNode.id, selectedOperationType, selectedSpecificType, parameters);
        }      
        // Apply specific type change if different
        if (selectedSpecificType && selectedSpecificType !== selectedNode.data.type) {
            updateNodeType(selectedNode.id, selectedOperationType, selectedSpecificType, parameters);
        }
        // Apply parameter changes
        Object.entries(parameters).forEach(([key, value]) => {
            updateNodeParameter(selectedNode.id, key, value);
        });
        // Reset pending changes
        setHasPendingChanges(false);
    };

    // Helper function to get current node definition
    const getCurrentNodeDefinition = () => {
        if (!selectedOperationType || !selectedClass || !selectedSpecificType) return null;
        
        let dataSource;
        switch (selectedOperationType) {
            case "Layer":
                dataSource = defaultLayers;
                break;
            case "TensorOp":
                dataSource = defaultTensorOps;
                break;
            case "Activator":
                dataSource = defaultActivators;
                break;
            case "Input":
                dataSource = defaultInputs;
                break;
            default:
                return null;
        }
        
        return dataSource?.data?.[selectedClass]?.[selectedSpecificType] || null;
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
                    value={selectedOperationType}
                    onChange={(e) => handleOperationTypeChange(e.target.value)}
                    fullWidth
                    size="small"
                    sx={{ mb: 2 }}
                >   
                        <MenuItem key={"Layer"} value={"Layer"}>{"Layer"}</MenuItem>
                        <MenuItem key={"TensorOp"} value={"TensorOp"}>{"Tensor Operation"}</MenuItem>
                        <MenuItem key={"Activator"} value={"Activator"}>{"Activator"}</MenuItem>
                        <MenuItem key={"Input"} value={"Input"}>{"Input"}</MenuItem>
                </TextField>

                {selectedOperationType && (
                    <TextField
                        select
                        label="Class"
                        value={selectedClass}
                        onChange={(e) => handleClassChange(e.target.value)}
                        fullWidth
                        size="small"
                        sx={{ mb: 2 }}
                        disabled={!selectedOperationType}
                    >   
                        {getAvailableClasses(selectedOperationType).map((className) => (
                            <MenuItem key={className} value={className}>{className}</MenuItem>
                        ))}
                    </TextField>
                )}

                {selectedClass && (
                    <TextField
                        select
                        label="Specific Type"
                        value={selectedSpecificType}
                        onChange={(e) => handleSpecificTypeChange(e.target.value)}
                        fullWidth
                        size="small"
                        sx={{ mb: 2 }}
                        disabled={!selectedClass}
                    >   
                        {getAvailableSpecificTypes(selectedOperationType, selectedClass).map((specificType) => (
                            <MenuItem key={specificType} value={specificType}>{specificType}</MenuItem>
                        ))}
                    </TextField>
                )}
                
                {selectedSpecificType && parameters && (
                    <ParameterInputs
                        operationType={selectedOperationType as "Layer" | "TensorOp" | "Activator" | "Input"}
                        nodeType={selectedSpecificType}
                        parameters={parameters}
                        onParameterChange={handleParameterChangeWithPending}
                        onValidationChange={handleValidationChange}
                        nodeDefinition={getCurrentNodeDefinition()}
                    />
                )}
                
                <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
                    <Button 
                        variant="contained" 
                        fullWidth 
                        onClick={handleApplyEdit}
                        disabled={!hasPendingChanges || hasValidationErrors}
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
                        Apply Edit
                    </Button>
                    <Button 
                        variant="contained" 
                        fullWidth 
                        onClick={deleteNodeLocal}
                        sx={{
                            backgroundColor: "#d32f2f",
                            borderRadius: '8px',
                            textTransform: 'none',
                            fontWeight: 600,
                            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                            '&:hover': {
                                backgroundColor: "#c62828",
                                boxShadow: '0 4px 8px rgba(0,0,0,0.2)'
                            }
                        }}
                    >
                        Delete
                    </Button>
                </Box>
            </Box>
    );
}
