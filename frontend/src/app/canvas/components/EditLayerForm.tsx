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

    // Helper function to get the first item from hierarchical data
    const getFirstItemFromData = (data: any) => {
        if (!data || !data.data) return null;
        
        for (const [className, classItems] of Object.entries(data.data)) {
            for (const [itemType, itemData] of Object.entries(classItems as any)) {
                return {
                    type: itemType,
                    class: className,
                    ...(itemData as any)
                };
            }
        }
        return null;
    };

    // Initialize state when selectedNode changes
    useEffect(() => {
        if (selectedNode) {
            setSelectedOperationType(selectedNode.data.operationType || "");
            setSelectedSpecificType(selectedNode.data.type || "");
            // Always load fresh parameters from the node (this will be the updated parameters after apply)
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
        
        // Load default parameters for the new operation type's first available option
        const newDefault = 
            newOperationType === "Layer" ? getFirstItemFromData(defaultLayers) :
            newOperationType === "TensorOp" ? getFirstItemFromData(defaultTensorOps) :
            newOperationType === "Activator" ? getFirstItemFromData(defaultActivators) : null;
        
        if (newDefault) {
            updateParameters(newDefault.parameters || {});
        }
        
        setHasPendingChanges(true);
    };

    // Handle class change
    const handleClassChange = (newClass: string) => {
        setSelectedClass(newClass);
        setSelectedSpecificType("");
        
        // Load parameters for the first type in the new class
        const data = 
            selectedOperationType === "Layer" ? defaultLayers :
            selectedOperationType === "TensorOp" ? defaultTensorOps :
            selectedOperationType === "Activator" ? defaultActivators : null;
        
        if (data?.data?.[newClass]) {
            const firstType = Object.keys(data.data[newClass])[0];
            const typeData = data.data[newClass][firstType];
            updateParameters(typeData?.parameters || {});
        }
        
        setHasPendingChanges(true);
    };

    // Handle specific type change
    const handleSpecificTypeChange = (newSpecificType: string) => {
        setSelectedSpecificType(newSpecificType);
        
        // Load default parameters for the new specific type
        const data = 
            selectedOperationType === "Layer" ? defaultLayers :
            selectedOperationType === "TensorOp" ? defaultTensorOps :
            selectedOperationType === "Activator" ? defaultActivators : null;
        
        if (data?.data?.[selectedClass]?.[newSpecificType]) {
            const typeData = data.data[selectedClass][newSpecificType];
            updateParameters(typeData?.parameters || {});
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
        
        const hasOperationTypeChange = selectedOperationType && selectedOperationType !== selectedNode.data.operationType;
        const hasSpecificTypeChange = selectedSpecificType && selectedSpecificType !== selectedNode.data.type;
        
        // Apply operation type change if different
        if (hasOperationTypeChange) {
            updateNodeOperationType(selectedNode.id, selectedOperationType);
        }
        
        // Apply specific type change if different (and no operation type change)
        else if (hasSpecificTypeChange) {
            updateNodeType(selectedNode.id, selectedOperationType, selectedSpecificType);
        }
        
        // Only apply parameter changes if no type changes occurred
        // (Type changes automatically set default parameters)
        else {
            Object.entries(parameters).forEach(([key, value]) => {
                updateNodeParameter(selectedNode.id, key, value);
            });
        }
        
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

