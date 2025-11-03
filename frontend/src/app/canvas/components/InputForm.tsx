"use client";

import { useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import MenuItem from "@mui/material/MenuItem";
import Button from "@mui/material/Button";
import { generateUniqueNodeId } from "../utils/idGenerator";
import { createNode } from "./TorchNodeCreator";

interface Props {
  nodes: any[];
  setNodes: (val: any) => void;
  updateNodeParameter: (elementID: string, parameterKey: string, parameterValue: any) => void 
  updateNodeType: (elementID: string, operationType: string, newtype: string) => void;
  updateNodeOperationType: (elementID: string, newOperationType: string) => void;
  deleteNode: (elementID: string) => void;
  getDefaults: () => any; // for editing within a node (TN)
}

export default function InputForm({ nodes, setNodes, updateNodeParameter, updateNodeType, updateNodeOperationType, deleteNode, getDefaults}: Props) {
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
    const newNode = createNode(
      newId,
      nodes.length, // posModifier
      "Input", // label
      "Input", // operation type
      "Input", // type
      {shapeType, dims: dims.map(Number),}, // parameters
      updateNodeParameter,
      updateNodeType,
      updateNodeOperationType,
      deleteNode,
      getDefaults
    )
    setNodes([
      ...nodes,
      newNode
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