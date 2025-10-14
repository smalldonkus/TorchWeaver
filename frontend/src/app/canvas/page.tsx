"use client"; // Enables React Server Components with client-side interactivity

import { useState, useCallback, useEffect } from "react";
import Box from "@mui/material/Box";
import CssBaseline from "@mui/material/CssBaseline";
import { applyNodeChanges, applyEdgeChanges, addEdge, OnSelectionChangeFunc} from "@xyflow/react";

import { initialNodes, initialEdges } from "./utils/constants";
import { Main, DrawerHeader } from "./utils/styled";
import AppBarHeader from "./components/AppBarHeader";
import Sidebar from "./components/Sidebar";
import Canvas from "./components/Canvas";
import useExport from "./hooks/useExport";
import defaultLayersJSON from "./utils/JsonDefaults/Layers.json"
import defaultTensorOpsJSON from "./utils/JsonDefaults/TensorOperations.json"
import defaultActivatorsJSON from "./utils/JsonDefaults/Activators.json"


// Main page component for the canvas feature
export default function CanvasPage() {

  // DEFAULTS: TODO: move
  const [defaultLayers, setDefaultLayers] = useState([]);
  useEffect(() => {
    setDefaultLayers(defaultLayersJSON.data);
  }, []);
  const [defaultTensorOps, setDefaultTensorOps] = useState([]);
  useEffect(() => {
    setDefaultTensorOps(defaultTensorOpsJSON.data);
  }, []);
  const [defaultActivators, setDefaultActivators] = useState([]);
  useEffect(() => {
    setDefaultActivators(defaultActivatorsJSON.data);
  }, []);

  // const [defaultLayers, setDefaultLayers] = useState([]);
  // useEffect(() => {
  //   const fetchFile = async () => {
  //     const jsonData = await fetch("./utils/JsonDefaults/Layers.json").then(res => res.json).then(data => data.data)
  //     setDefaultLayers(jsonData)
  //   }
  // }, []);
  // console.log(defaultLayers)

  // State to control if the sidebar is open
  const [open, setOpen] = useState(true);
  // State for the nodes in the canvas
  const [nodes, setNodes] = useState(initialNodes);
  // State for the edges (connections) in the canvas
  const [edges, setEdges] = useState(initialEdges);
  // State for which menu is selected in the sidebar
  const [selectedMenu, setSelectedMenu] = useState("Layers");
  // state for the currently selected Nodes, only the first used currently
  const[selectedNodes, setSelectedNodes] = useState<Node[]>([])
  // shows selected edges, not currently used
  const[selectedEdges, setSelectedEdges] = useState<Edge[]>([])

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

  const onSelectionChange: OnSelectionChangeFunc = useCallback(
    ({nodes, edges}) => {
      setSelectedNodes((nodes));
      setSelectedEdges((edges));
    },[]
  );

  const updateNodeParameter = (elementID, parameterKey, parameterValue) => {
    setNodes(oldNodes =>
      oldNodes.map(e => e.id == elementID ? {...e, data: {...e.data, parameters : {...e.data.parameters, [parameterKey] : parameterValue}}} : e)
    )
    setSelectedNodes(oldNodes =>
      oldNodes.map(e => e.id == elementID ? {...e, data: {...e.data, parameters : {...e.data.parameters, [parameterKey] : parameterValue}}} : e)
    )
  }

  const updateNodeLabel = (elementID, newLabel) => {
    setNodes(oldNodes =>
      oldNodes.map(e => e.id == elementID ? {...e, data: {...e.data, label: newLabel}} : e)
    )
    setSelectedNodes(oldNodes =>
      oldNodes.map(e => e.id == elementID ? {...e, data: {...e.data, label: newLabel}} : e)
    )
  }

  const updateNodeType = (elementID, operationType, newtype) => {
    const newDefault = 
      operationType === "Layer" ? defaultLayers.find(e => newtype == e.type) :
      operationType === "TensorOp" ? defaultTensorOps.find(e => newtype == e.type) :
      operationType === "Activator" ? defaultActivators.find(e => newtype == e.type) : null;
      setNodes(oldNodes =>
      oldNodes.map(e => e.id == elementID ? {...e, data: {...e.data, type: newtype, parameters : newDefault.parameters}} : e)
    )
    setSelectedNodes(oldNodes =>
      oldNodes.map(e => e.id == elementID ? {...e, data: {...e.data, type: newtype, parameters : newDefault.parameters}} : e)
    )
  }

  const updateNodeOperationType = (elementID, newOperationType) => {
    const newDefault = 
      newOperationType === "Layer" ? defaultLayers[0] :
      newOperationType === "TensorOp" ? defaultTensorOps[0] :
      newOperationType === "Activator" ? defaultActivators[0] : null;
    setNodes(oldNodes =>
      oldNodes.map(e => e.id === elementID ? {...e, data: {...e.data, type: newDefault.type, 
        operationType: newOperationType, parameters: newDefault.parameters}} : e)
    );
    setSelectedNodes(oldNodes =>
      oldNodes.map(e => e.id === elementID ? {...e, data: {...e.data, type: newDefault.type, 
        operationType: newOperationType, parameters: newDefault.parameters}} : e)
    );
  }

  const deleteNode = (elementID) => {
    setNodes(oldNodes =>
      oldNodes.filter((e) => e.id != elementID)
    );
    setSelectedNodes(oldNodes =>
      oldNodes.filter((e) => e.id != elementID)
    );
  }

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
        selectedNodes={selectedNodes}
        updateNodeLabel={updateNodeLabel}
        updateNodeType={updateNodeType}
        updateNodeOperationType={updateNodeOperationType}
        updateNodeParameter={updateNodeParameter}
        deleteNode={deleteNode}
        defaultLayers={defaultLayers}
        defaultTensorOps={defaultTensorOps}
        defaultActivators={defaultActivators}
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
          onSelectionChange={onSelectionChange}
        />
      </Main>
    </Box>
  );
}
