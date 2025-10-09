"use client";
import Box from "@mui/material/Box";
import { ReactFlow } from "@xyflow/react";
import "@xyflow/react/dist/style.css";

interface Props {
  nodes: any[];
  edges: any[];
  onNodesChange: any;
  onEdgesChange: any;
  onConnect: any;
}

export default function Canvas({
  nodes,
  edges,
  onNodesChange,
  onEdgesChange,
  onConnect,
}: Props) {
  return (
    <Box
      sx={{
        width: "100%",
        height: "80vh",
        background: "#f9fafb",
        borderRadius: 2,
        boxShadow: 1,
      }}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        fitView
      />
    </Box>
  );
}
