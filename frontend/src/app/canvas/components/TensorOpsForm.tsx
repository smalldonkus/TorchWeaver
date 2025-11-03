"use client";

import { useState, useEffect } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import MenuItem from "@mui/material/MenuItem";
import Button from "@mui/material/Button";
import { generateUniqueNodeId } from "../utils/idGenerator";
import ParameterInputs from "./ParameterInputs";
import { useParameterHandling } from "../hooks/useParameterHandling";
import { createNode } from "./TorchNodeCreator";

interface Props {
  nodes: any[];
  setNodes: (val: any) => void;
  defaultTensorOps: any; // Changed from any[] to any to handle new structure

  // for TorchNode functionality, allows it to update itself (TN)
    updateNodeParameter: (elementID: string, parameterKey: string, parameterValue: any) => void 
    updateNodeType: (elementID: string, operationType: string, newtype: string) => void;
    updateNodeOperationType: (elementID: string, newOperationType: string) => void;
    deleteNode: (elementID: string) => void;
    getDefaults: () => any; // for editing within a node (TN)
}

export default function TensorOpsForm({ nodes, setNodes, defaultTensorOps, updateNodeParameter, updateNodeType, updateNodeOperationType, deleteNode, getDefaults }: Props) {
  // Use parameter handling hook
  const { 
    parameters, 
    hasValidationErrors, 
    handleParameterChange, 
    handleValidationChange, 
    updateParameters 
  } = useParameterHandling();

  const [selectedClass, setSelectedClass] = useState<string>("");
  const [selectedOpType, setSelectedOpType] = useState<string>("");
  const [chosenOp, setChosenOp] = useState<any>(null);

  // Extract global classes from the new structure
  const globalClasses = defaultTensorOps?.data ? Object.keys(defaultTensorOps.data) : [];

  // Initialize selections when data loads
  useEffect(() => {
    if (globalClasses.length > 0) {
      const firstClass = globalClasses[0];
      const firstClassOps = defaultTensorOps.data[firstClass];
      const firstOpType = Object.keys(firstClassOps)[0];
      
      setSelectedClass(firstClass);
      setSelectedOpType(firstOpType);
      setChosenOp({
        class: firstClass,
        type: firstOpType,
        ...firstClassOps[firstOpType]
      });
      updateParameters(firstClassOps[firstOpType]?.parameters || {});
    }
  }, [defaultTensorOps, updateParameters]);

  if (!defaultTensorOps || !defaultTensorOps.data || Object.keys(defaultTensorOps.data).length === 0) {
    return <div>Loading tensor operations...</div>;
  }

  function handleClassChange(className: string) {
    setSelectedClass(className);
    const classOps = defaultTensorOps.data[className];
    const firstOpType = Object.keys(classOps)[0];
    setSelectedOpType(firstOpType);
    const newOp = {
      class: className,
      type: firstOpType,
      ...classOps[firstOpType]
    };
    setChosenOp(newOp);
    updateParameters(newOp?.parameters || {});
  }

  function handleOpTypeChange(opType: string) {
    setSelectedOpType(opType);
    const opData = defaultTensorOps.data[selectedClass][opType];
    const newOp = {
      class: selectedClass,
      type: opType,
      ...opData
    };
    setChosenOp(newOp);
    updateParameters(newOp?.parameters || {});
  }

  const addTensorOp = () => {
    if (hasValidationErrors) {
      alert("Please fix parameter errors before adding the tensor operation.");
      return;
    }

    const newId = generateUniqueNodeId("tensorop", nodes);
    const newNode = createNode(
        newId,
        nodes.length, // posModifier
        chosenOp.type, // label
        "TensorOp", // operation type
        chosenOp.type, // type
        parameters,
        updateNodeParameter,
        updateNodeType,
        updateNodeOperationType,
        deleteNode,
        getDefaults
    );
    setNodes([
      ...nodes,
      newNode
    ]);
    
    // Reset to first selection after adding
    if (globalClasses.length > 0) {
      const firstClass = globalClasses[0];
      const firstClassOps = defaultTensorOps.data[firstClass];
      const firstOpType = Object.keys(firstClassOps)[0];
      
      setSelectedClass(firstClass);
      setSelectedOpType(firstOpType);
      setChosenOp({
        class: firstClass,
        type: firstOpType,
        ...firstClassOps[firstOpType]
      });
      updateParameters(firstClassOps[firstOpType]?.parameters || {});
    }
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="subtitle1" sx={{ mb: 2 }}>
        Add Tensor Operation
      </Typography>
      
      {/* Dropdown to select operation class */}
      <TextField
        select
        label="Operation Class"
        value={selectedClass}
        onChange={e => handleClassChange(e.target.value)}
        fullWidth
        size="small"
        sx={{ mb: 2 }}
      >
        {globalClasses.map((className) => (
          <MenuItem key={className} value={className}>{className}</MenuItem>
        ))}
      </TextField>

      {/* Dropdown to select specific operation type within the class */}
      {selectedClass && (
        <TextField
          select
          label="Operation Type"
          value={selectedOpType}
          onChange={e => handleOpTypeChange(e.target.value)}
          fullWidth
          size="small"
          sx={{ mb: 2 }}
        >
          {Object.keys(defaultTensorOps.data[selectedClass]).map((opType) => (
            <MenuItem key={opType} value={opType}>{opType}</MenuItem>
          ))}
        </TextField>
      )}
      
      {chosenOp && (
        <ParameterInputs
          operationType="TensorOp"
          nodeType={chosenOp.type}
          parameters={parameters}
          onParameterChange={handleParameterChange}
          onValidationChange={handleValidationChange}
        />
      )}
      
      <Button variant="contained" fullWidth onClick={addTensorOp}>
        Add Tensor Operation
      </Button>
    </Box>
  );
}