"use client";

import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import { generateUniqueNodeId } from "../utils/idGenerator";

interface Props {
  nodes: any[];
  setNodes: (val: any) => void;
}

export default function OutputForm({ nodes, setNodes }: Props) {
  const addOutput = () => {
    const newId = generateUniqueNodeId("output", nodes);
    setNodes([
      ...nodes,
      {
        id: newId,
        position: { x: 300, y: 100 + nodes.length * 60 },
        data: {
          label: "Output",
          operationType: "Output",
          parameters: {},
          outgoing_edges_count: 0 // Initialize with 0 outgoing edges (outputs typically have 0)
        },
      },
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