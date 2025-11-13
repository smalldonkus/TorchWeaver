"use client";
import React, { useCallback } from 'react';
import Box from "@mui/material/Box";
import { ReactFlow, Background, Controls } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useEdgeReconnection } from '../hooks/useEdgeReconnection';
import ErrorsButton from './ErrorsButton';
import ExportButton from './ExportButton';
import SaveButton from './SaveButton';

interface Props {
  nodes: any[];
  edges: any[];
  nodeTypes: any;
  onNodesChange: any;
  onEdgesChange: any;
  onConnect: any;
  onSelectionChange: any;
  setEdges: (edges: any[] | ((prevEdges: any[]) => any[])) => void;
  handleExport?: () => void;
  handleSave?: () => void;
  openErrorBox?: boolean;
  setOpenErrorBox?: (val: boolean) => void;
}

export default function Canvas({
  nodes,
  edges,
  nodeTypes,
  onNodesChange,
  onEdgesChange,
  onConnect,
  onSelectionChange,
  setEdges,
  handleExport,
  handleSave,
  openErrorBox,
  setOpenErrorBox
}: Props) {
  const { onReconnectStart, onReconnect, onReconnectEnd } = useEdgeReconnection(setEdges);

  return (
    <Box
      sx={{
        width: "100%",
        height: "80vh",
        background: "#ffffff",
        borderRadius: 2,
        boxShadow: 1,
        position: "relative",
      }}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onSelectionChange={onSelectionChange}
        onReconnect={onReconnect}
        onReconnectStart={onReconnectStart}
        onReconnectEnd={onReconnectEnd}
        snapToGrid={false}
        fitView
        style={{ backgroundColor: "#f7f9fb" }}
        attributionPosition="top-right"
      >
        <Controls />
        <Background color="#aaa" gap={16} />
      </ReactFlow>
      {/* Fixed action buttons */}
      {setOpenErrorBox && (
        <ErrorsButton openErrorBox={openErrorBox} setOpenErrorBox={setOpenErrorBox} />
      )}
      {handleExport && (
        <ExportButton handleExport={handleExport} />
      )}
      {handleSave && (
        <SaveButton handleSave={handleSave} />
      )}
    </Box>
  );
}
