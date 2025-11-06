"use client"; // Enables React Server Components with client-side interactivity

import { useState, useCallback, useEffect } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import CssBaseline from "@mui/material/CssBaseline";
import { applyNodeChanges, applyEdgeChanges, addEdge, OnSelectionChangeFunc, Node, Edge, MarkerType} from "@xyflow/react";

import { initialNodes, initialEdges } from "./utils/constants";
import { Main, DrawerHeader } from "./utils/styled";
import AppBarHeader from "./components/AppBarHeader";
import Sidebar from "./components/Sidebar";
import Canvas from "./components/Canvas";
import useExport from "./hooks/useExport";
import useOperationDefinitions from "./hooks/useOperationDefinitions";
import { generateUniqueNodeId } from "./utils/idGenerator";
import { determineCanInheritFromParent, linkParametersToChannels } from "./components/TorchNodeCreator";
import { propagateChannelInheritance, findNodeDefinition, handleInheritFromParentChange } from "./utils/channelPropagation";

import useParse from "./hooks/useParse";
import ErrorBox from "./components/ErrorBox";

import TorchNode from "./components/TorchNode";

// Main page component for the canvas feature
export default function CanvasPage() {

  // Fetch operation definitions from backend
  const { layers: defaultLayers, tensorOps: defaultTensorOps, activators: defaultActivators, inputs: defaultInputs, loading: operationsLoading, error: operationsError } = useOperationDefinitions();

  // State to control if the sidebar is open
  const [open, setOpen] = useState(true);
  // State for the nodes in the canvas
  const [nodes, setNodes] = useState<any[]>(initialNodes);
  // State for the edges (connections) in the canvas
  const [edges, setEdges] = useState<any[]>(initialEdges);

  const nodeTypes = {
    torchNode : TorchNode
  };
  
  // Add arrows to all existing edges on component mount
  useEffect(() => {
    if (edges.length > 0) {
      const edgesWithArrows = edges.map(edge => ({
        ...edge,
        markerEnd: {
          type: MarkerType.Arrow,
        },
      }));
      setEdges(edgesWithArrows);
    }
  }, []); // Only run once on mount

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
      // Add arrow marker to the new edge
      const edgeWithArrow = {
        ...params,
        markerEnd: {
          type: MarkerType.Arrow,
        },
      };
      const newEdges = addEdge(edgeWithArrow, edges);
      setEdges(newEdges);
      
      // Update outgoing edge counts for affected nodes
      updateOutgoingEdgeCounts(newEdges);
    },
    [edges]
  );

  useEffect(() => {
    updateOutgoingEdgeCounts(edges); // was creating errors in delete nodes (do not know why, hopefully this doesn't break anything)
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
      
      // Update node styling based on selection
      const selectedNodeIds = nodes.map(node => node.id);
      setNodes((currentNodes) =>
        currentNodes.map(node => ({
          ...node,
          style: {
            ...node.style,
            border: selectedNodeIds.includes(node.id) 
              ? '1px solid #3b71e4ff'  // Blue border for selected nodes
              : '1px solid #e5e7eb'  // Default gray border for unselected nodes
          }
        }))
      );
    },[]
  );

  const updateNodeParameter = (elementID: string, parameterKey: string, parameterValue: any) => {
    
    setNodes((oldNodes: any[]) => {
      const updatedNodes = oldNodes.map(e => {
        if (e.id === elementID) {
          // Update the parameter
          const updatedParameters = {...(e.data.parameters || {}), [parameterKey]: parameterValue};
          
          // Get node definition to check for channel editing capabilities
          const nodeDefinition = findNodeDefinition(
            e.data.operationType, 
            e.data.type,
            defaultLayers,
            defaultTensorOps,
            defaultActivators,
            defaultInputs
          );
          
          let updatedData = {
            ...e.data, 
            parameters: updatedParameters
          };
          
          // Check if this node can edit channels and if the changed parameter affects channels
          if (nodeDefinition?.parseCheck?.CanEditChannels && e.data.operationType !== "Output") {
            const channelLinks = nodeDefinition.parseCheck.ChannelLinks || [];
            
            // Check if the changed parameter is linked to input or output channels
            channelLinks.forEach((link: any) => {
              if (link.inputParam === parameterKey) {
                // Update inputChannels
                updatedData.inputChannels = parameterValue;
              }
              if (link.outputParam === parameterKey) {
                // Update outputChannels  
                updatedData.outputChannels = parameterValue;

                // Trigger propagation after this update completes
                setTimeout(() => {
                  setEdges((currentEdges) => {
                    setNodes((currentNodes) => {
                      // Propagate channel inheritance to children
                      const propagatedNodes = propagateChannelInheritance(
                        elementID,
                        parameterValue,
                        currentNodes,
                        currentEdges,
                        defaultLayers,
                        defaultTensorOps,
                        defaultActivators,
                        defaultInputs
                      );
                      return propagatedNodes;
                    });
                    return currentEdges; // Don't modify edges
                  });
                }, 0);

              }
            });
            
            // Also update can_inherit_from_parent if inherit_from_parent parameter changed
            if (parameterKey === 'inherit_from_parent') {
              updatedData.can_inherit_from_parent = parameterValue;

              // When inherit_from_parent is set to true, inherit from parent node
              if (parameterValue === true) {
                handleInheritFromParentChange(
                  elementID,
                  setEdges,
                  setNodes,
                  defaultLayers,
                  defaultTensorOps,
                  defaultActivators,
                  defaultInputs
                );
              }
            }
          }
          
          return {...e, data: updatedData};
        }
        return e;
      });
      
      return updatedNodes;
    });
  }

  // Helper function to find a type in the new hierarchical structure
  const findTypeInData = (data: any, targetType: string) => {
    if (!data || !data.data) return null;
    
    for (const [className, classItems] of Object.entries(data.data)) {
      for (const [itemType, itemData] of Object.entries(classItems as any)) {
        if (itemType === targetType) {
          return {
            type: itemType,
            class: className,
            ...(itemData as any)
          };
        }
      }
    }
    return null;
  };

  const updateNodeType = (elementID: string, operationType: string, newType: string, newParameters: any) => {

    const newDefault = 
      operationType === "Layer" ? findTypeInData(defaultLayers, newType) :
      operationType === "TensorOp" ? findTypeInData(defaultTensorOps, newType) :
      operationType === "Activator" ? findTypeInData(defaultActivators, newType) : null;
    
    if (!newDefault) return;
    
    // Generate new ID based on operation type
    const operationPrefix = 
      operationType === "Layer" ? "layer" :
      operationType === "TensorOp" ? "tensorop" :
      operationType === "Activator" ? "activator" : "node";
    
    const newNodeId = generateUniqueNodeId(operationPrefix, nodes);
      
    // Update nodes with new ID and properties
    setNodes((oldNodes: any[]) =>
      oldNodes.map(e => e.id === elementID ? {...e, 
        id: newNodeId,
        data: {...e.data, type: newType, label: newType, parameters : newParameters || {}}} : e)
    );
    
    // Update selected nodes
    setSelectedNodes((oldNodes: any[]) =>
      oldNodes.map(e => e.id === elementID ? {...e, 
        id: newNodeId,
        data: {...e.data, type: newType, label: newType, parameters : newParameters || {}}} : e)
    );
    
    // Update all edges that reference the old node ID
    setEdges((oldEdges: any[]) =>
      oldEdges.map(edge => ({
        ...edge,
        source: edge.source === elementID ? newNodeId : edge.source,
        target: edge.target === elementID ? newNodeId : edge.target
      }))
    );
  }

  // Helper function to get the first item from hierarchical data
  const getFirstItemFromData = (data: any) => {
    if (!data || !data.data) return null;
    
    for (const [className, classItems] of Object.entries(data.data)) {
      for (const [itemType, itemData] of Object.entries(classItems as any)) {
        return {
          type: itemType,
          class: className,
          ...(itemData as any)
        };
      }
    }
    return null;
  };

  const updateNodeOperationType = (elementID: string, newOperationType: string, newSpecificType: string, newParameters: any) => {

    const newDefault = 
      newOperationType === "Layer" ? findTypeInData(defaultLayers, newSpecificType) :
      newOperationType === "TensorOp" ? findTypeInData(defaultTensorOps, newSpecificType) :
      newOperationType === "Activator" ? findTypeInData(defaultActivators, newSpecificType) : null;
    
    if (!newDefault) return;
    
    // Generate new ID based on operation type
    const operationPrefix = 
      newOperationType === "Layer" ? "layer" :
      newOperationType === "TensorOp" ? "tensorop" :
      newOperationType === "Activator" ? "activator" : "node";
    
    const newNodeId = generateUniqueNodeId(operationPrefix, nodes);
      
    // Update nodes with new ID and properties
    setNodes((oldNodes: any[]) =>
      oldNodes.map(e => e.id === elementID ? {...e, 
        id: newNodeId,
        data: {...e.data, type: newDefault.type, label: newDefault.type,
        operationType: newOperationType, parameters: newParameters || {}}} : e)
    );
    
    // Update selected nodes
    setSelectedNodes((oldNodes: any[]) =>
      oldNodes.map(e => e.id === elementID ? {...e, 
        id: newNodeId,
        data: {...e.data, type: newDefault.type, label: newDefault.type,
        operationType: newOperationType, parameters: newParameters || {}}} : e)
    );
    
    // Update all edges that reference the old node ID
    setEdges((oldEdges: any[]) =>
      oldEdges.map(edge => ({
        ...edge,
        source: edge.source === elementID ? newNodeId : edge.source,
        target: edge.target === elementID ? newNodeId : edge.target
      }))
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
    // remove edges from node
    setEdges (oldEdges =>
      oldEdges.filter((e) => !(e.source === elementID || e.target === elementID))
    );
  };

  // Custom hook to handle exporting the current canvas state
  const handleExport = useExport(nodes, edges, defaultLayers, defaultTensorOps, defaultActivators);

  const unpackErrorIds = (errors: any[]) => {
    const rtn: any[] = [];
    errors.forEach((value) => {
      if (value.flaggedNodes != null && value.flaggedNodes.length != 0) {
        value.flaggedNodes.forEach((v: any) => rtn.push(v));
      }
    })
    return rtn;
  }

  const getAllNodeErrors = (errors: any[], eID) => {
    const rtn: any[] = [];
    errors.forEach((v) => {
      if (v.flaggedNodes != null && v.flaggedNodes.length != 0){
        if (v.flaggedNodes.includes(eID)) {rtn.push(v.errorMsg)};
      }
    });
    return rtn;
  }

  // updates state variables when errors are added
  useEffect( () => {
      seterrorOpen(errors.length == 0 ? false : true);
      seterrorMsgs(errors.map((e) => e.errorMsg));

      // fills/updates/clears the errors array of each node, if of type: "torchNode"
      setNodes((oldNodes) =>
        oldNodes.map((e) => {
          // find all errors associated with this node
          const errorsMsgArr = getAllNodeErrors(errors, e.id);
          return {
            ...e,
            data: {
              ...e.data,
              errors : errorsMsgArr
            }            
          }
        })
      );

      // set the border of each "error'd" node to red
      const errorIDs: any[] = unpackErrorIds(errors);
      setNodes((oldNodes) => 
        oldNodes.map((e) => errorIDs.includes(e.id) ? 
          {...e, style: {...e.style, border: "1px solid #d32f2f"}}
          :
          {...e, style: {...e.style, border: "1px solid black"}})
      );
  }, [errors]);

  // Separate useEffect for debugging - only triggers when nodes change
  // useEffect(() => {
  //   // Debug: Log all nodes and their hidden parameters
  //   console.log("=== NODES DEBUG ===");
  //   nodes.forEach(node => {
  //     if (node.data.operationType !== "Output") {
  //       console.log(`Node: ${node.id} (${node.data.type})`);
  //       console.log(`  inputChannels: ${node.data.inputChannels}`);
  //       console.log(`  outputChannels: ${node.data.outputChannels}`);
  //       console.log(`  can_inherit_from_parent: ${node.data.can_inherit_from_parent}`);
  //       console.log(`  inherit_from_parent param: ${node.data.parameters?.inherit_from_parent}`);
  //       console.log(`  parameters:`, node.data.parameters);
  //       console.log("---");
  //     }
  //   });
  //   console.log("=== END DEBUG ===");
  // }, [nodes]);

  const getSetters = () => {
    return {
      updateNodeParameter     : updateNodeParameter,
      updateNodeType          : updateNodeType,
      updateNodeOperationType : updateNodeOperationType,
      deleteNode              : deleteNode
    }
  }

  const getDefaults = () => {
    return {
      defaultActivators: defaultActivators, 
      defaultTensorOps: defaultTensorOps, 
      defaultLayers: defaultLayers,
      defaultInputs: defaultInputs
    }
  }

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
        getSetters={getSetters}
        getDefaults={getDefaults}
        defaultLayers={defaultLayers}
        defaultTensorOps={defaultTensorOps}
        defaultActivators={defaultActivators}
        defaultInputs={defaultInputs}
      />
      {/* Main content area for the canvas */}
      <Main open={open}>
        <DrawerHeader /> {/* Spacer for the header */}
        {/* Canvas component where nodes and edges are displayed and edited */}
        <Canvas
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onSelectionChange={onSelectionChange}
          setEdges={setEdges}
        />
      </Main>
      <ErrorBox key={"errorBox"} isOpen={openErrorBox} setOpen={setOpenErrorBox} messages={errorMsgs}/>
    </Box>
  );
}
