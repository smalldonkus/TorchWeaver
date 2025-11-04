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
export default function EditLayerForm({selectedNodes, defaultActivators, defaultTensorOps, defaultLayers, updateNodeType, updateNodeOperationType, updateNodeParameter , deleteNode}: Props) {
    
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
    }, [selectedNode, updateParameters]);

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
    }, [selectedOperationType, selectedSpecificType, defaultLayers, defaultTensorOps, defaultActivators]);

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
                        operationType={selectedOperationType as "Layer" | "TensorOp" | "Activator"}
                        nodeType={selectedSpecificType}
                        parameters={parameters}
                        onParameterChange={handleParameterChangeWithPending}
                        onValidationChange={handleValidationChange}
                    />
                )}
                
                <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
                    <Button 
                        variant="contained" 
                        fullWidth 
                        onClick={handleApplyEdit}
                        disabled={!hasPendingChanges || hasValidationErrors}
                        sx={{ backgroundColor: 'primary.main' }}
                    >
                        Apply Edit
                    </Button>
                    <Button 
                        variant="contained" 
                        fullWidth 
                        style={{backgroundColor: "red"}} 
                        onClick={deleteNodeLocal}
                    >
                        Delete
                    </Button>
                </Box>
            </Box>
    );
}

