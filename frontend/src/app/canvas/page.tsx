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
import useSave from "./hooks/useSave";

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
  
  // Load saved network if available
  useEffect(() => {
    async function fetchNetwork() {
    const params = new URLSearchParams(window.location.search);
    // support either 'id' or 'network_id' query param (some code used network_id)
    const id = params.get("id") || params.get("network_id");
      if (!id) {
        console.log("No network ID in URL");
        return;
      }

      try {
        console.log("=== LOADING NETWORK ===");
        console.log("Fetching network ID:", id);
        
        const response = await fetch(`http://localhost:5000/load_network?id=${id}`);
        const data = await response.json();

        console.log("Raw response data:", data);

        if (data.network) {
          // Detailed structure validation
          console.log("Network data structure validation:");
          console.log("- Has nodes array:", Array.isArray(data.network.nodes));
          console.log("- Has edges array:", Array.isArray(data.network.edges));
          
          if (data.network.nodes && data.network.nodes.length > 0) {
            console.log("Sample node structure:", data.network.nodes[0]);
          }
          
          if (Array.isArray(data.network.nodes) && Array.isArray(data.network.edges)) {
            console.log("Setting network state with:");
            console.log("- Nodes:", data.network.nodes.length);
            console.log("- Edges:", data.network.edges.length);

            // Ensure edges include markerEnd for arrows and nodes have expected fields
            const normalizedEdges = data.network.edges.map((edge: any) => ({
              ...edge,
              markerEnd: edge.markerEnd || { type: MarkerType.Arrow }
            }));

            // Add getSetters and getDefaults functions to loaded nodes (they're not serialized)
            const normalizedNodes = data.network.nodes.map((node: any) => ({
              ...node,
              data: {
                ...node.data,
                getSetters: getSetters,
                getDefaults: getDefaults
              }
            }));

            setNodes(normalizedNodes);
            setEdges(normalizedEdges);
          } else {
            console.error("Invalid network structure:", data.network);
          }
        } else {
          console.error("No network data in response for ID:", id);
        }
        console.log("=== END LOADING ===");
      } catch (err) {
        console.error("Error loading network:", err);
      }
    }
    fetchNetwork();
  }, []);  // Add arrows to all existing edges on component mount


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

      // Handle inheritance when connection is made
      setTimeout(() => {
        setEdges((currentEdges) => {
          setNodes((currentNodes) => {
            // find the target node of the new connection
            const targetNode = currentNodes.find(node => node.id === params.target);
            // if the target node can inherit from parent, update its parameters
            if (targetNode && targetNode.data.can_inherit_from_parent) {
              // Find the source node
              const sourceNode = currentNodes.find(node => node.id === params.source);
              
              if (sourceNode) {
                
                const nodeDefinition = findNodeDefinition(
                  targetNode.data.operationType,
                  targetNode.data.type,
                  defaultLayers,
                  defaultTensorOps,
                  defaultActivators,
                  defaultInputs
                );
                
                // Will need to include the new edge for max calculation
                const edgesIncludingNew = [...currentEdges, { source: params.source, target: params.target }];
              
              const updatedNodes = currentNodes.map(node => {
                if (node.id === params.target) {
                  let inheritedData = { ...node.data };
                  
                  const canEditChannels = nodeDefinition?.parseCheck?.CanEditChannels;
                  
                  if (!canEditChannels) {
                    // Pass-through node with potentially multiple parents: use max output channels
                    const maxParentChannels = getMaxParentOutputChannels(params.target, currentNodes, edgesIncludingNew);
                    if (maxParentChannels !== null) {
                      inheritedData.inputChannels = maxParentChannels;
                      inheritedData.outputChannels = maxParentChannels;
                    }
                  } else {
                    // Editable node: use single parent's output
                    inheritedData.inputChannels = sourceNode.data.outputChannels;
                    
                    // Update linked parameters
                    const channelLinks = nodeDefinition?.parseCheck?.ChannelLinks || [];
                    channelLinks.forEach((link: any) => {
                      if (link.inputParam) {
                        inheritedData.parameters = {
                          ...inheritedData.parameters,
                          [link.inputParam]: sourceNode.data.outputChannels
                        };
                      }
                    });
                  }
                  
                  return { ...node, data: inheritedData };
                }
                return node;
              });
              
              const updatedTargetNode = updatedNodes.find(node => node.id === params.target);
              
              if (updatedTargetNode) {
                setTimeout(() => {
                  setEdges((currentEdges) => {
                    setNodes((propagationNodes) => {
                      return propagateChannelInheritance(
                        params.target,
                        updatedTargetNode.data.outputChannels,
                        propagationNodes,
                        currentEdges,
                        defaultLayers,
                        defaultTensorOps,
                        defaultActivators,
                        defaultInputs
                      );
                    });
                    return currentEdges;
                  });
                }, 0);
              }
              
              return updatedNodes;
            }
          }
          
          return currentNodes;
          });
          return currentEdges;
        });
      }, 0);
    },
    [edges, defaultLayers, defaultTensorOps, defaultActivators, defaultInputs]
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

  // Helper function to get max output channels from all parent nodes
  const getMaxParentOutputChannels = (nodeId: string, currentNodes: any[], currentEdges: any[]): number | null => {
    // Find all parent edges (edges pointing to this node)
    const parentEdges = currentEdges.filter(edge => edge.target === nodeId);
    
    if (parentEdges.length === 0) return null;
    
    // Get all parent nodes' output channels
    const parentOutputChannels = parentEdges
      .map(edge => {
        const parentNode = currentNodes.find(node => node.id === edge.source);
        return parentNode?.data.outputChannels;
      })
      .filter(channels => channels !== undefined && channels !== null);
    
    if (parentOutputChannels.length === 0) return null;
    
    // Return the maximum
    return Math.max(...parentOutputChannels);
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
                console.log("Updating inputChannels due to parameter change");
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
      operationType === "Input" ? findTypeInData(defaultInputs, newType) :
      operationType === "Layer" ? findTypeInData(defaultLayers, newType) :
      operationType === "TensorOp" ? findTypeInData(defaultTensorOps, newType) :
      operationType === "Activator" ? findTypeInData(defaultActivators, newType) : null;
    
    if (!newDefault) return;
    
    // Use setTimeout to access current state
    setTimeout(() => {
      setEdges((currentEdges) => {
        setNodes((currentNodes) => {
          // Generate new ID based on operation type
          const operationPrefix = 
            operationType === "Input" ? "input" :
            operationType === "Layer" ? "layer" :
            operationType === "TensorOp" ? "tensorop" :
            operationType === "Activator" ? "activator" : "node";
          
          const newNodeId = generateUniqueNodeId(operationPrefix, currentNodes);
          
          // Calculate hidden attributes for the new type
          let channelData = linkParametersToChannels(newDefault, newParameters || {});
          const canInherit = determineCanInheritFromParent(newDefault, newParameters || {});
          
          // Check if node should inherit from parent
          if (canInherit) {
            const canEditChannels = newDefault?.parseCheck?.CanEditChannels;
            
            if (!canEditChannels) {
              // Pass-through node with potentially multiple parents: use max output channels
              const maxParentChannels = getMaxParentOutputChannels(elementID, currentNodes, currentEdges);
              if (maxParentChannels !== null) {
                channelData.inputChannels = maxParentChannels;
                channelData.outputChannels = maxParentChannels;
              }
            } else {
              // Editable node: use single parent's output
              const parentEdge = currentEdges.find(edge => edge.target === elementID);
              if (parentEdge) {
                const parentNode = currentNodes.find(node => node.id === parentEdge.source);
                if (parentNode && parentNode.data.outputChannels !== undefined) {
                  // Update input channels to match parent's output
                  channelData.inputChannels = parentNode.data.outputChannels;
                  
                  // Update linked input parameter
                  const channelLinks = newDefault?.parseCheck?.ChannelLinks || [];
                  const inputLink = channelLinks.find((link: any) => link.inputParam);
                  if (inputLink && newParameters) {
                    newParameters[inputLink.inputParam] = parentNode.data.outputChannels;
                    // Recalculate with updated parameters
                    channelData = linkParametersToChannels(newDefault, newParameters);
                  }
                }
              }
            }
          }
          
          // Update nodes with new ID and properties
          const updatedNodes = currentNodes.map(e => e.id === elementID ? {...e, 
            id: newNodeId,
            data: {
              ...e.data, 
              type: newType, 
              label: newType, 
              parameters: newParameters || {},
              inputChannels: channelData.inputChannels,
              outputChannels: channelData.outputChannels,
              can_inherit_from_parent: canInherit
            }} : e);
          
          // Update selected nodes
          setSelectedNodes((oldNodes: any[]) =>
            oldNodes.map(e => e.id === elementID ? {...e, 
              id: newNodeId,
              data: {...e.data, type: newType, label: newType, parameters : newParameters || {}}} : e)
          );
          
          // Update edges with new node ID
          setEdges((oldEdges: any[]) =>
            oldEdges.map(edge => ({
              ...edge,
              source: edge.source === elementID ? newNodeId : edge.source,
              target: edge.target === elementID ? newNodeId : edge.target
            }))
          );
          
          // Propagate new output channels to children
          setTimeout(() => {
            setEdges((edges) => {
              setNodes((nodes) => {
                return propagateChannelInheritance(
                  newNodeId,
                  channelData.outputChannels,
                  nodes,
                  edges,
                  defaultLayers,
                  defaultTensorOps,
                  defaultActivators,
                  defaultInputs
                );
              });
              return edges;
            });
          }, 0);
          
          return updatedNodes;
        });
        return currentEdges;
      });
    }, 0);
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
    // Simply delegate to updateNodeType which handles everything including inheritance
    updateNodeType(elementID, newOperationType, newSpecificType, newParameters);
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

  const handleSave = useSave(nodes, edges);

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
        handleSave={handleSave}
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
