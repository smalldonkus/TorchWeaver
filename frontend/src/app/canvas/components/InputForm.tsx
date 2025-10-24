"use client";

import { useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import MenuItem from "@mui/material/MenuItem";
import Button from "@mui/material/Button";
import { generateUniqueNodeId } from "../utils/idGenerator";

interface Props {
  nodes: any[];
  setNodes: (val: any) => void;
}

export default function InputForm({ nodes, setNodes }: Props) {
  const [shapeType, setShapeType] = useState("1D");
  const [dims, setDims] = useState<string[]>([""]);

  // Update dims array when shapeType changes
  const handleShapeTypeChange = (type: string) => {
    setShapeType(type);
    if (type === "1D") setDims([""]);
    if (type === "2D") setDims(["", ""]);
    if (type === "3D") setDims(["", "", ""]);
  };

  // Update a specific dimension
  const handleDimChange = (idx: number, value: string) => {
    setDims(dims.map((d, i) => (i === idx ? value : d)));
  };

  const addInput = () => {
    const newId = generateUniqueNodeId("input", nodes);
    setNodes([
      ...nodes,
      {
        id: newId,
        position: { x: 50, y: 100 + nodes.length * 60 },
        data: {
          label: "Input",
          operationType: "Input",
          parameters: {
            shapeType,
            dims: dims.map(Number),
          },
        },
      },
    ]);
    setShapeType("1D");
    setDims([""]);
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="subtitle1" sx={{ mb: 2 }}>
        Add Input
      </Typography>
      <TextField
        select
        label="Shape Type"
        value={shapeType}
        onChange={e => handleShapeTypeChange(e.target.value)}
        fullWidth
        size="small"
        sx={{ mb: 2 }}
      >
        <MenuItem value="1D">1D</MenuItem>
        <MenuItem value="2D">2D</MenuItem>
        <MenuItem value="3D">3D</MenuItem>
      </TextField>
      {dims.map((dim, idx) => (
        <TextField
          key={idx}
          label={`Dimension ${idx + 1}`}
          value={dim}
          onChange={e => handleDimChange(idx, e.target.value)}
          type="number"
          fullWidth
          size="small"
          sx={{ mb: 2 }}
        />
      ))}
      <Button variant="contained" fullWidth onClick={addInput}>
        Add Input
      </Button>
    </Box>
  );
}