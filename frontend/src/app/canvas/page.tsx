"use client"; // Enables React Server Components with client-side interactivity

import { useState, useCallback, useEffect } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import CssBaseline from "@mui/material/CssBaseline";
import { applyNodeChanges, applyEdgeChanges, addEdge, OnSelectionChangeFunc, Node, Edge} from "@xyflow/react";

import { initialNodes, initialEdges } from "./utils/constants";
import { Main, DrawerHeader } from "./utils/styled";
import AppBarHeader from "./components/AppBarHeader";
import Sidebar from "./components/Sidebar";
import Canvas from "./components/Canvas";
import useExport from "./hooks/useExport";
import useOperationDefinitions from "./hooks/useOperationDefinitions";

import useParse from "./hooks/useParse";
import ErrorBox from "./components/ErrorBox";

// Main page component for the canvas feature
export default function CanvasPage() {

  // Fetch operation definitions from backend
  const { layers: defaultLayers, tensorOps: defaultTensorOps, activators: defaultActivators, loading: operationsLoading, error: operationsError } = useOperationDefinitions();

  // State to control if the sidebar is open
  const [open, setOpen] = useState(true);
  // State for the nodes in the canvas
  const [nodes, setNodes] = useState<any[]>(initialNodes);
  // State for the edges (connections) in the canvas
  const [edges, setEdges] = useState<any[]>(initialEdges);
  // State for which menu is selected in the sidebar
  const [selectedMenu, setSelectedMenu] = useState("Layers");
  // state for the currently selected Nodes, only the first used currently
  const [selectedNodes, setSelectedNodes] = useState<Node[]>([])
  // shows selected edges, not currently used
  const [selectedEdges, setSelectedEdges] = useState<Edge[]>([])

  // used for logging errors
  const [errors, setErrors] = useState<any[]>([]);
  // error UI variables
  const [errorOpen, seterrorOpen] = useState(false);
  const [errorMsgs, seterrorMsgs] = useState<any[]>([]);
  // used for opening the error drawer
  const [openErrorBox, setOpenErrorBox] = useState(false);

  // Handler for when nodes are changed (moved, edited, etc.)
  const onNodesChange = useCallback(
    // (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
    // []
    (changes) => {
      const newNodes = applyNodeChanges(changes, nodes);
      setNodes(newNodes);
      // Update outgoing edge counts for affected nodes

      // update nodes when nodes update
      useParse(nodes, edges).then((e) => {setErrors(e)});
    },
    [nodes]
  );
  // Handler for when edges are changed (added, removed, etc.)
  const onEdgesChange = useCallback(
    (changes) => {
      const newEdges = applyEdgeChanges(changes, edges);
      setEdges(newEdges);
      
      // Update outgoing edge counts for affected nodes
      updateOutgoingEdgeCounts(newEdges);
    },
    [edges]
  );
  
  // Handler for when a new connection (edge) is made between nodes
  const onConnect = useCallback(
    (params) => {
      const newEdges = addEdge(params, edges);
      setEdges(newEdges);
      
      // Update outgoing edge counts for affected nodes
      updateOutgoingEdgeCounts(newEdges);
    },
    [edges]
  );

  useEffect(() => {
    useParse(nodes, edges).then((e) => {setErrors(e)});
  }, [edges]) // having this sit inside other functions causes issues

  // Function to update outgoing edge counts for all nodes
  const updateOutgoingEdgeCounts = (currentEdges: any[]) => {
    // Build outgoing edges map
    const outgoingEdges: Record<string, string[]> = {};
    currentEdges.forEach((edge) => {
      if (!outgoingEdges[edge.source]) outgoingEdges[edge.source] = [];
      outgoingEdges[edge.source].push(edge.target);
    });

    // Update nodes with new outgoing edge counts
    setNodes((currentNodes) =>
      currentNodes.map((node) => ({
        ...node,
        data: {
          ...node.data,
          outgoing_edges_count: (outgoingEdges[node.id] || []).length
        }
      }))
    );
  };

  const onSelectionChange: OnSelectionChangeFunc = useCallback(
    ({nodes, edges}) => {
      setSelectedNodes((nodes));
      setSelectedEdges((edges));
    },[]
  );

  const updateNodeParameter = (elementID: string, parameterKey: string, parameterValue: any) => {
    setNodes((oldNodes: any[]) =>
      oldNodes.map(e => e.id === elementID ? {...e, data: {...e.data, parameters : {...(e.data.parameters || {}), [parameterKey] : parameterValue}}} : e)
    );
    setSelectedNodes((oldNodes: any[]) =>
      oldNodes.map(e => e.id === elementID ? {...e, data: {...e.data, parameters : {...(e.data.parameters || {}), [parameterKey] : parameterValue}}} : e)
    );
  }

  const updateNodeType = (elementID: string, operationType: string, newtype: string) => {
    const newDefault = 
      operationType === "Layer" ? defaultLayers.find(e => newtype === e.type) :
      operationType === "TensorOp" ? defaultTensorOps.find(e => newtype === e.type) :
      operationType === "Activator" ? defaultActivators.find(e => newtype === e.type) : null;
    
    if (!newDefault) return;
      
    setNodes((oldNodes: any[]) =>
      oldNodes.map(e => e.id === elementID ? {...e, data: {...e.data, type: newtype, label: newtype, parameters : newDefault.parameters || {}}} : e)
    )
    setSelectedNodes((oldNodes: any[]) =>
      oldNodes.map(e => e.id === elementID ? {...e, data: {...e.data, type: newtype, label: newtype, parameters : newDefault.parameters || {}}} : e)
    )
  }

  const updateNodeOperationType = (elementID: string, newOperationType: string) => {
    const newDefault = 
      newOperationType === "Layer" ? defaultLayers[0] :
      newOperationType === "TensorOp" ? defaultTensorOps[0] :
      newOperationType === "Activator" ? defaultActivators[0] : null;
    
    if (!newDefault) return;
      
    setNodes((oldNodes: any[]) =>
      oldNodes.map(e => e.id === elementID ? {...e, data: {...e.data, type: newDefault.type, 
        operationType: newOperationType, parameters: newDefault.parameters || {}}} : e)
    );
    setSelectedNodes((oldNodes: any[]) =>
      oldNodes.map(e => e.id === elementID ? {...e, data: {...e.data, type: newDefault.type, 
        operationType: newOperationType, parameters: newDefault.parameters || {}}} : e)
    );
  }

  const deleteNode = (elementID: string) => {
    // Remove the node from nodes state
    setNodes(oldNodes =>
      oldNodes.filter((e) => e.id !== elementID)
    );
    
    // Remove the node from selected nodes
    setSelectedNodes(oldNodes =>
      oldNodes.filter((e) => e.id !== elementID)
    );
    
    // Remove all edges connected to this node (both incoming and outgoing)
    const newEdges = edges.filter((edge) => edge.source !== elementID && edge.target !== elementID);
    setEdges(newEdges);
    
    // Update outgoing edge counts for remaining nodes
    updateOutgoingEdgeCounts(newEdges);
  }

  // Custom hook to handle exporting the current canvas state
  const handleExport = useExport(nodes, edges, defaultLayers, defaultTensorOps, defaultActivators);


  const unpackErrorIds = (errors: any[]) => {
    const rtn = [];
    errors.forEach((value) => {
      if (value.flaggedNodes != null && value.flaggedNodes.length != 0) {
        value.flaggedNodes.forEach((v) => rtn.push(v));
      }
    })
    return rtn;
  }

  const inErrorColour = "#d32f2f"
  const stdColour = "#ffffff"
  // updates state variables when errors are added
  useEffect( () => {
      seterrorOpen(errors.length == 0 ? false : true);
      seterrorMsgs(errors.map((e) => e.errorMsg));
      const errorIDs: any[] = unpackErrorIds(errors);
      console.log(errorIDs)
      setNodes((oldNodes) => 
        oldNodes.map((e) => errorIDs.includes(e.id) ? 
          {...e, style: {...e.style, background: inErrorColour}}
          :
          {...e, style: {...e.style, background: stdColour}})
      );
  }, [errors]);


  // Show loading state while fetching operations
  if (operationsLoading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", alignItems: "center", height: "100vh" }}>
        <Typography>Loading operation definitions...</Typography>
      </Box>
    );
  }

  // Show error state if operations failed to load
  if (operationsError) {
    return (
      <Box sx={{ display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", height: "100vh" }}>
        <Typography color="error" sx={{ mb: 2 }}>Failed to load operation definitions</Typography>
        <Typography variant="body2">{operationsError}</Typography>
        <Typography variant="body2" sx={{ mt: 1 }}>Please ensure the backend server is running on http://localhost:5000</Typography>
      </Box>
    );
  }

  // Render the main layout
  return (
    <Box sx={{ display: "flex" }}>
      <CssBaseline /> {/* Resets CSS for consistent styling */}
      {/* Top app bar/header */}
      <AppBarHeader open={open} setOpen={setOpen} openErrorBox={openErrorBox} setOpenErrorBox={setOpenErrorBox}/>
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
      <ErrorBox key={"errorBox"} isOpen={openErrorBox} setOpen={setOpenErrorBox} messages={errorMsgs}/>
    </Box>
  );
}
