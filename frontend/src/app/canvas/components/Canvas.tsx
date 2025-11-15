"use client";
import React, { useCallback } from 'react';
import Box from "@mui/material/Box";
import { ReactFlow, Background, Controls, useReactFlow } from "@xyflow/react";
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
  OnEdgesDelete: any;
  onConnect: any;
  onSelectionChange: any;
  setEdges: (edges: any[] | ((prevEdges: any[]) => any[])) => void;
  handleExport?: () => void;
  handleSave?: () => void;
  errorMessages?: string[];
}

export default function Canvas({
  nodes,
  edges,
  nodeTypes,
  onNodesChange,
  onEdgesChange,
  OnEdgesDelete,
  onConnect,
  onSelectionChange,
  setEdges,
  handleExport,
  handleSave,
  errorMessages
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
        onEdgesDelete={OnEdgesDelete}
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
      <ErrorsButton errorMessages={errorMessages || []} />
      {handleExport && (
        <ExportButton handleExport={handleExport} />
      )}
      {handleSave && (
        <SaveButton handleSave={handleSave} />
      )}
    </Box>
  );
}
