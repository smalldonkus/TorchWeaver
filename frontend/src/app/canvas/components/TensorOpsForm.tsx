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

interface Props {
  nodes: any[];
  setNodes: (val: any) => void;
  defaultTensorOps: any[];
}

export default function TensorOpsForm({ nodes, setNodes, defaultTensorOps }: Props) {
  // Use parameter handling hook
  const { 
    parameters, 
    hasValidationErrors, 
    handleParameterChange, 
    handleValidationChange, 
    updateParameters 
  } = useParameterHandling();

  const [chosenOp, setChosenOp] = useState(defaultTensorOps?.[0] || null);

  // Update parameters when chosen operation changes
  useEffect(() => {
    if (defaultTensorOps && defaultTensorOps.length > 0) {
      setChosenOp(defaultTensorOps[0]);
      updateParameters(defaultTensorOps[0]?.parameters || {});
    }
  }, [defaultTensorOps, updateParameters]);

  // Redundant checks for loading and error states
  if (!defaultTensorOps || defaultTensorOps.length === 0) {
    return <div>Loading tensor operations...</div>;
  }

  function setOp(opType: string) {
    const newOp = defaultTensorOps.find((op) => op.type === opType);
    setChosenOp(newOp);
    updateParameters(newOp?.parameters || {});
  }

  const addTensorOp = () => {
    if (hasValidationErrors) {
      alert("Please fix parameter errors before adding the tensor operation.");
      return;
    }

    const newId = generateUniqueNodeId("tensorop", nodes);
    setNodes([
      ...nodes,
      {
        id: newId,
        position: { x: 200, y: 100 + nodes.length * 60 },
        data: {
          label: chosenOp.type,
          operationType: "TensorOp",
          type: chosenOp.type,
          parameters: parameters,
          outgoing_edges_count: 0
        },
      },
    ]);
    setChosenOp(defaultTensorOps[0]);
    updateParameters(defaultTensorOps[0]?.parameters || {});
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="subtitle1" sx={{ mb: 2 }}>
        Add Tensor Operation
      </Typography>
      <TextField
        select
        label="Operation Type"
        value={chosenOp?.type || ""}
        onChange={e => setOp(e.target.value)}
        fullWidth
        size="small"
        sx={{ mb: 2 }}
      >
        {defaultTensorOps.map((op) => (
          <MenuItem key={op.type} value={op.type}>{op.type}</MenuItem>
        ))}
      </TextField>
      
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