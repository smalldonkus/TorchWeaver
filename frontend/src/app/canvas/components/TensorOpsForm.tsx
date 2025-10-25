"use client";

import { useState, useEffect } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import MenuItem from "@mui/material/MenuItem";
import Button from "@mui/material/Button";
import { generateUniqueNodeId } from "../utils/idGenerator";

interface Props {
  nodes: any[];
  setNodes: (val: any) => void;
  defaultTensorOps: any[];
}

export default function TensorOpsForm({ nodes, setNodes, defaultTensorOps }: Props) {
  if (!defaultTensorOps || defaultTensorOps.length === 0) {
    return <div>Loading tensor operations...</div>;
  }

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
          parameters: chosenOp.parameters,
          outgoing_edges_count: 0 // Initialize with 0 outgoing edges
        },
      },
    ]);
    setChosenOp(defaultTensorOps[0]);
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="subtitle1" sx={{ mb: 2 }}>
        Add Tensor Operation
      </Typography>
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