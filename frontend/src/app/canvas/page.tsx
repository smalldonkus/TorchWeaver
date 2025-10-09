"use client"; // Enables React Server Components with client-side interactivity

import { useState, useCallback } from "react";
import Box from "@mui/material/Box";
import CssBaseline from "@mui/material/CssBaseline";
import { applyNodeChanges, applyEdgeChanges, addEdge } from "@xyflow/react";

import { initialNodes, initialEdges } from "./utils/constants";
import { Main, DrawerHeader } from "./utils/styled";
import AppBarHeader from "./components/AppBarHeader";
import Sidebar from "./components/Sidebar";
import Canvas from "./components/Canvas";
import useExport from "./hooks/useExport";

// Main page component for the canvas feature
export default function CanvasPage() {
  // State to control if the sidebar is open
  const [open, setOpen] = useState(true);
  // State for the nodes in the canvas
  const [nodes, setNodes] = useState(initialNodes);
  // State for the edges (connections) in the canvas
  const [edges, setEdges] = useState(initialEdges);
  // State for which menu is selected in the sidebar
  const [selectedMenu, setSelectedMenu] = useState("Layers");

  // Handler for when nodes are changed (moved, edited, etc.)
  const onNodesChange = useCallback(
    (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
    []
  );
  // Handler for when edges are changed (added, removed, etc.)
  const onEdgesChange = useCallback(
    (changes) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    []
  );
  // Handler for when a new connection (edge) is made between nodes
  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    []
  );

  // Custom hook to handle exporting the current canvas state
  const handleExport = useExport(nodes, edges);

  // Render the main layout
  return (
    <Box sx={{ display: "flex" }}>
      <CssBaseline /> {/* Resets CSS for consistent styling */}
      {/* Top app bar/header */}
      <AppBarHeader open={open} setOpen={setOpen} />
      {/* Sidebar with menu and export functionality */}
      <Sidebar
        open={open}
        setOpen={setOpen}
        selectedMenu={selectedMenu}
        setSelectedMenu={setSelectedMenu}
        nodes={nodes}
        setNodes={setNodes}
        handleExport={handleExport}
      />
      {/* Main content area for the canvas */}
      <Main open={open}>
        <DrawerHeader /> {/* Spacer for the header */}
        {/* Canvas component where nodes and edges are displayed and edited */}
        <Canvas
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
        />
      </Main>
    </Box>
  );
}
