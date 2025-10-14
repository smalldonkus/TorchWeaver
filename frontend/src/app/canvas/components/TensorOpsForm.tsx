"use client";

import { useState, useEffect } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import MenuItem from "@mui/material/MenuItem";
import Button from "@mui/material/Button";

interface Props {
  nodes: any[];
  setNodes: (val: any) => void;
  defaultTensorOps: any[];
}

export default function TensorOpsForm({ nodes, setNodes, defaultTensorOps }: Props) {
  if (!defaultTensorOps || defaultTensorOps.length === 0) {
    return <div>Loading tensor operations...</div>;
  }

  const [newLabel, setNewLabel] = useState("");
  const [chosenOp, setChosenOp] = useState(defaultTensorOps[0]);

  useEffect(() => {
    setChosenOp(defaultTensorOps[0]);
  }, [defaultTensorOps]);

  function setOp(opType: string) {
    setChosenOp(defaultTensorOps.find((op) => op.type === opType));
  }

  const updateParam = (paramKey: string, paramValue: string) => {
    setChosenOp(prev => ({
      ...prev,
      parameters: { ...prev.parameters, [paramKey]: paramValue }
    }));
  };

  const addTensorOp = () => {
    const newId = `tensorop${nodes.length + 1}`;
    setNodes([
      ...nodes,
      {
        id: newId,
        position: { x: 200, y: 100 + nodes.length * 60 },
        data: {
          label: `${chosenOp.type}: ${newLabel || `TensorOp ${nodes.length + 1}`}`,
          operationType: "TensorOp",
          type: chosenOp.type,
          parameters: chosenOp.parameters
        },
      },
    ]);
    setNewLabel("");
    setChosenOp(defaultTensorOps[0]);
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="subtitle1" sx={{ mb: 2 }}>
        Add Tensor Operation
      </Typography>
      <TextField
        label="Tensor Op label"
        value={newLabel}
        onChange={e => setNewLabel(e.target.value)}
        fullWidth
        size="small"
        sx={{ mb: 2 }}
      />
      <TextField
        select
        label="Operation Type"
        value={chosenOp?.type}
        onChange={e => setOp(e.target.value)}
        fullWidth
        size="small"
        sx={{ mb: 2 }}
      >
        {defaultTensorOps.map((op) => (
          <MenuItem key={op.type} value={op.type}>{op.type}</MenuItem>
        ))}
      </TextField>
      {chosenOp && chosenOp.parameters &&
        Object.keys(chosenOp.parameters).map((paramKey, i) => (
          <TextField
            key={i}
            label={paramKey}
            value={chosenOp.parameters[paramKey]}
            onChange={e => updateParam(paramKey, e.target.value)}
            type={typeof chosenOp.parameters[paramKey] === "number" ? "number" : "text"}
            fullWidth
            size="small"
            sx={{ mb: 2 }}
          />
        ))
      }
      <Button variant="contained" fullWidth onClick={addTensorOp}>
        Add Tensor Operation
      </Button>
    </Box>
  );
}