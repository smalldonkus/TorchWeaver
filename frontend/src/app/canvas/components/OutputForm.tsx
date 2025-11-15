"use client";

import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import { generateUniqueNodeId } from "../utils/idGenerator";
import { createNode } from "./TorchNodeCreator";

interface Props {
  nodes: any[];
  addNode: (val: any) => void;
  getSetters: () => any;
  getDefaults: () => any; // for editing within a node (TN)
}

export default function OutputForm({ nodes, addNode, getSetters, getDefaults }: Props) {
  const addOutput = () => {
    const newId = generateUniqueNodeId("output", nodes);
    const newNode = createNode(
      newId,
      nodes.length, // posModifier
      "Output", // label
      "Output", // operation type
      "Output", // type
      {}, // parameters
      getSetters,
      getDefaults
    )
    addNode([
      ...nodes,
      newNode
    ]);
  };

  return (
    <Box sx={{ p: 2 }}>
      <Button variant="contained" fullWidth onClick={addOutput}>
        Add Output
      </Button>
    </Box>
  );
}