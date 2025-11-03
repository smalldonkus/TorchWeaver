"use client";

import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
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

export default function OutputForm({ nodes, setNodes, updateNodeParameter, updateNodeType, updateNodeOperationType, deleteNode, getDefaults }: Props) {
  const addOutput = () => {
    const newId = generateUniqueNodeId("output", nodes);
    const newNode = createNode(
      newId,
      nodes.length, // posModifier
      "Output", // label
      "Output", // operation type
      "Output", // type
      {}, // parameters
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
  };

  return (
    <Box sx={{ p: 2 }}>
      <Button variant="contained" fullWidth onClick={addOutput}>
        Add Output
      </Button>
    </Box>
  );
}