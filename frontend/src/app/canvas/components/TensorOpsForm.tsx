"use client";

import { useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import MenuItem from "@mui/material/MenuItem";
import Button from "@mui/material/Button";

interface Props {
  nodes: any[];
  setNodes: (val: any) => void;
}

export default function TensorOpsForm({ nodes, setNodes }: Props) {
  const [opType, setOpType] = useState("Add");
  const [label, setLabel] = useState("");

  const addTensorOp = () => {
    const newId = `n${nodes.length + 1}`;
    setNodes([
      ...nodes,
      {
        id: newId,
        position: { x: 200, y: 100 + nodes.length * 60 },
        data: {
          label: `${opType}: ${label || `TensorOp ${nodes.length + 1}`}`,
          operationType: "TensorOp",
          opType,
        },
      },
    ]);
    setLabel("");
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="subtitle1" sx={{ mb: 2 }}>
        Add Tensor Operation
      </Typography>
      <TextField
        select
        label="Operation Type"
        value={opType}
        onChange={e => setOpType(e.target.value)}
        fullWidth
        size="small"
        sx={{ mb: 2 }}
      >
        <MenuItem value="MaxPool2d">MaxPool2d</MenuItem>
        <MenuItem value="LocalResponseNorm">LocalResponseNorm</MenuItem>
        <MenuItem value="Dropout">Dropout</MenuItem>
        {/* Add more tensor operations as needed */}
      </TextField>
      <TextField
        label="Label"
        value={label}
        onChange={e => setLabel(e.target.value)}
        fullWidth
        size="small"
        sx={{ mb: 2 }}
      />
      <Button variant="contained" fullWidth onClick={addTensorOp}>
        Add Tensor Operation
      </Button>
    </Box>
  );
}