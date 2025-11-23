"use client"; // Enables React Server Components with client-side interactivity

import { useState, useCallback, useEffect, useRef, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import CssBaseline from "@mui/material/CssBaseline";
import Snackbar from "@mui/material/Snackbar";
import Alert from "@mui/material/Alert";
import { applyNodeChanges, applyEdgeChanges, addEdge, OnSelectionChangeFunc, Node, Edge, MarkerType, ReactFlowProvider } from "@xyflow/react";

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
import * as htmlToImage from 'html-to-image';
import { toPng } from 'html-to-image';

import useParse from "./hooks/useParse";
import useSave from "./hooks/useSave";

import TorchNode from "./components/TorchNode";
import UnsavedChangesDialog from "./components/UnsavedChangesDialog";
import { stringify } from "querystring";

// Main page component for the canvas feature
function CanvasPageContent() {

  // Fetch operation definitions from backend
  const { layers: defaultLayers, tensorOps: defaultTensorOps, activators: defaultActivators, inputs: defaultInputs, loading: operationsLoading, error: operationsError } = useOperationDefinitions();

  // State to control if the sidebar is open
  const [open, setOpen] = useState(true);
  // State for the nodes in the canvas
  const [nodes, setNodes] = useState<any[]>(initialNodes);
  // State for the edges (connections) in the canvas
  const [edges, setEdges] = useState<any[]>(initialEdges);
  // State for the name of network
  const [name, setName] = useState<string>("Untitled");
  // Track last saved state to detect unsaved changes
  const [lastSavedState, setLastSavedState] = useState<{nodes: any[], edges: any[], name: string} | null>(null);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  // Dialog state for navigation warning
  const [showUnsavedDialog, setShowUnsavedDialog] = useState(false);
  const [pendingNavigation, setPendingNavigation] = useState<string | null>(null);
  const router = useRouter();

  const nodeTypes = {
    torchNode : TorchNode
  };

  const canvasRef = useRef<HTMLDivElement>(null) //ref to canvas to save images
  
  const searchParams = useSearchParams();

  // Load saved network if available
  useEffect(() => {
    async function fetchNetwork() {
    // support either 'id' or 'network_id' query param (some code used network_id)
    const id = searchParams.get("id") || searchParams.get("network_id");
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

          if (data.network.name) {
            setName(data.network.name);
          }
          
          if (data.network.nodes && data.network.nodes.length > 0) {
            console.log("Sample node structure:", data.network.nodes[0]);
          }
          
          if (Array.isArray(data.network.nodes) && Array.isArray(data.network.edges)) {
            console.log("Setting network state with:");
            console.log("- Nodes:", data.network.nodes.length);
            console.log("- Edges:", data.network.edges.length);

            // Wait for defaults to load before processing nodes
            if (!defaultLayers || !defaultTensorOps || !defaultActivators || !defaultInputs) {
              return;
            }

            // Ensure edges include markerEnd for arrows and nodes have expected fields
            const normalizedEdges = data.network.edges.map((edge: any) => ({
              ...edge,
              style: { strokeWidth: 2, ...edge.style },
              markerEnd: edge.markerEnd || { type: MarkerType.Arrow, width: 16, height: 16 }
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
            // Initialize undo list with loaded state
            setUndoList([{n: normalizedNodes, e: normalizedEdges}]);
            setUndoListIndex(0);
            // Set initial saved state after loading
            setLastSavedState({
              nodes: normalizedNodes,
              edges: normalizedEdges,
              name: data.network.name || "Untitled"
            });
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
  }, [defaultLayers, defaultTensorOps, defaultActivators, defaultInputs]);  // Re-run when defaults are loaded


  useEffect(() => {
    if (edges.length > 0) {
      const edgesWithArrows = edges.map(edge => ({
        ...edge,
        style: { strokeWidth: 2, ...edge.style },
        markerEnd: {
          type: MarkerType.Arrow,
          width: 16,
          height: 16
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
  
  // Snackbar state for success/error messages
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'warning' | 'info';
  }>({
    open: false,
    message: '',
    severity: 'success'
  });

  const showSnackbar = (message: string, severity: 'success' | 'error' | 'warning' | 'info' = 'success') => {
    setSnackbar({ open: true, message, severity });
  };

  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };
  
  const [undoList, setUndoList] = useState<any[]>([{n : [], e: []}]);
  const [undoListIndex, setUndoListIndex] = useState<number>(0);
  
  /* CONCURRENCY PROTECTIONS */
  // reference to nodes and edges
  const nodesRef    = useRef(nodes);
  const edgesRef    = useRef(edges);
  const undoListRef = useRef(undoList);
  const undoListIndexRef = useRef(undoListIndex);

  // updates nodes and edges references
  useEffect(() => {nodesRef.current = nodes}, [nodes]);
  useEffect(() => {edgesRef.current = edges}, [edges]);
  useEffect(() => {undoListRef.current = undoList}, [undoList]);
  useEffect(() => {undoListIndexRef.current = undoListIndex}, [undoListIndex]);


  const handleSetUndoList = (n, e, reRenderers?) => {

    const currUndoList = undoListRef.current;
    const currIndex = undoListIndexRef.current;
    const appendObject = {
      n : n,
      e : e,
    };

    // console.log(currUndoList, currIndex, currIndex == currUndoList.length - 1);
    if (currIndex == -1) {
      setUndoListIndex(curr => curr + 1); // list is about to appended by one, so this works
      setUndoList([appendObject]);
    }

    else if (currIndex == currUndoList.length - 1) {
      setUndoListIndex(curr => curr + 1); // list is about to appended by one, so this works
      setUndoList((curr) => [
        ...curr,
        appendObject
      ]);
    }
    else if (currIndex < currUndoList.length - 1){
      const slicedList = currUndoList.slice(0, currIndex + 1);
      setUndoListIndex(curr => slicedList.length == 0 ? 0 : curr + 1);
      setUndoList([
        ...slicedList,
        appendObject
      ])
    }
    else {
      console.error("unindented functionality in undo");
    }
  };

  useEffect(() => {
    const s: string[] = undoList.map((e) => e.n.length == 0 ? "nLen: 0, eLen: 0" : ", nLen: " + e.n.length + ", eLen: " + e.e.length);
    console.log("undolist current: " + s.join(", "));
    const s2: string[] = undoList.map((e) => "head: " + (e.n.length == 0 ? "noNode" : e.n[e.n.length - 1].data.label));
    console.log("undolist current: " + s2.join(", "), ", uLI: " + undoListIndex);
    console.log(nodesRef.current.map((e) => e.id).join(", "));
    // if (undoList != undefined && undoList.length != 0 && undoList[undoListIndex].n.length != 0 && undoList[undoListIndex].n[0].data.label == "Conv2d"){
    //   console.log(undoList[undoListIndex].n[0].data.parameters, undoList[undoListIndex].n[0].selected);
    // }
  }, [undoList, undoListIndex]);


  const doUndo = () => {
    // if empty
    const currIndex = undoListIndexRef.current;

    if ((currIndex == 0)) {
      console.log("reached end of the undoList");
      return;
    };
    const currEra = undoListRef.current[currIndex - 1];

    setNodes(currEra.n);
    setEdges(currEra.e);

    // move the index one step towards the zeroth index
    setUndoListIndex(currIndex == -1 ? currIndex : currIndex - 1);
  }
  const doRedo = () => {
    const currUndoList = undoListRef.current;
    const currIndex = undoListIndexRef.current;

    if ((currIndex + 1 == currUndoList.length)) {
      console.log("Upto date");
      return;
    };
    const currEra = currUndoList[currIndex + 1];

    setNodes(currEra.n);
    setEdges(currEra.e);

    // move the index one step towards the ending index
    setUndoListIndex(currIndex + 1 == currUndoList.length ? currIndex : currIndex + 1);
  };

  const addNode = (toBeSetTo) => {
    handleSetUndoList(toBeSetTo, edgesRef.current);
    setNodes(toBeSetTo);
  }


  // Handler for when nodes are changed (moved, edited, etc.)
  const onNodesChange = useCallback(
    // (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
    // []
    (changes) => {
      const newNodes = applyNodeChanges(changes, nodes);
      setNodes(newNodes);

      if (changes != undefined && changes[0] != undefined && changes[0].type === "position"){
        setUndoList(lst => lst.map( // for each era in lst
          era => ({
            n : era.n.map( // for node in era
              (eraNode) => ({
                ...eraNode,
                position: changes[0].id === eraNode.id ? changes[0].position : eraNode.position
              })
            ),
            e : era.e
          })
        ));
      };
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
        style: { strokeWidth: 2 },
        markerEnd: {
          type: MarkerType.Arrow,
          width: 16,
          height: 16
        },
      };
      const newEdges = addEdge(edgeWithArrow, edges);
      setEdges(newEdges);

      // Update outgoing edge counts for affected nodes
      updateOutgoingEdgeCounts(newEdges);

      // strange functionality, but it cannot be avoided (at least i can't think how)
      handleSetUndoList(nodesRef.current, newEdges);

      // Handle inheritance when connection is made
      setTimeout(() => {
        setEdges((currentEdges) => {
          setNodes((currentNodes) => {
            // find the target node of the new connection
            const targetNode = currentNodes.find(node => node.id === params.target);
            // Check both the flag AND the parameter (parameter takes precedence if recently changed)
            const shouldInherit = targetNode?.data.can_inherit_from_parent || targetNode?.data.parameters?.inherit_from_parent;
            // if the target node can inherit from parent, update its parameters
            if (targetNode && shouldInherit) {
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
              
              // Apply propagation if needed, using the already-updated nodes
              let finalNodes = updatedNodes;
              if (updatedTargetNode) {
                finalNodes = propagateChannelInheritance(
                  params.target,
                  updatedTargetNode.data.outputChannels,
                  updatedNodes,  // Use updatedNodes, not a fresh state
                  currentEdges,
                  defaultLayers,
                  defaultTensorOps,
                  defaultActivators,
                  defaultInputs
                );
              }
              
              return finalNodes;
            }
            return currentNodes;
          }
          
          return currentNodes;
          });
          return currentEdges;
        });
      }, 0);
    },
    [edges, defaultLayers, defaultTensorOps, defaultActivators, defaultInputs]
  );

  const OnEdgesDelete = useCallback(
    (delEdges) => {

      if (delEdges.length == 0) {
        console.log("delEdges empty")
        return;
      }
      const localNodesRef = nodesRef.current;
      const localEdgesRef = edgesRef.current; // this shit changes like no-one's business.
      const trimmedEdges = localEdgesRef.filter((e) => !delEdges.some(e2 => e2.id === e.id));

      handleSetUndoList(localNodesRef, trimmedEdges);
    }, []
  );

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

  const updateNodeParameter = useCallback((elementID: string, parameterKey: string, parameterValue: any) => {
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
                // Check if input and output channel parameters are the same (linked)
                const channelLinks = nodeDefinition.parseCheck.ChannelLinks || [];
                let inputParam = null;
                let outputParam = null;
                
                channelLinks.forEach((link: any) => {
                  if (link.inputParam) inputParam = link.inputParam;
                  if (link.outputParam) outputParam = link.outputParam;
                });
                
                const areChannelsLinked = (inputParam && outputParam && inputParam === outputParam);
                
                // If channels are linked, we need to propagate after inheritance
                if (areChannelsLinked) {
                  // Trigger handleInheritFromParentChange and then propagate
                  setTimeout(() => {
                    setEdges((currentEdges) => {
                      setNodes((currentNodes) => {
                        // Find parent node
                        const parentEdge = currentEdges.find(edge => edge.target === elementID);
                        if (parentEdge) {
                          const parentNode = currentNodes.find(node => node.id === parentEdge.source);
                          if (parentNode && parentNode.data.outputChannels !== undefined) {
                            // Update the node with inherited values
                            const updatedNodes = currentNodes.map(node => {
                              if (node.id === elementID) {
                                const updatedParameters = {
                                  ...node.data.parameters
                                };
                                if (inputParam) {
                                  updatedParameters[inputParam] = parentNode.data.outputChannels;
                                }
                                
                                return {
                                  ...node,
                                  data: {
                                    ...node.data,
                                    inputChannels: parentNode.data.outputChannels,
                                    outputChannels: parentNode.data.outputChannels,
                                    parameters: updatedParameters
                                  }
                                };
                              }
                              return node;
                            });
                            
                            // Propagate the output channel changes to children
                            const propagatedNodes = propagateChannelInheritance(
                              elementID,
                              parentNode.data.outputChannels,
                              updatedNodes,
                              currentEdges,
                              defaultLayers,
                              defaultTensorOps,
                              defaultActivators,
                              defaultInputs
                            );
                            return propagatedNodes;
                          }
                        }
                        return currentNodes;
                      });
                      return currentEdges;
                    });
                  }, 0);
                } else {
                  // Channels not linked, just use regular inherit logic
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
          }
          
          return {...e, data: updatedData};
        }
        return e;
      });
      return updatedNodes;
    });
  }, [defaultLayers, defaultTensorOps, defaultActivators, defaultInputs]);

  const handleSetUndoListWhenUpdateNodeParameterIsCalled = (doReRender: () => void) => {
    setTimeout(() => handleSetUndoList(nodesRef.current, edgesRef.current), 0);
  }
  // useEffect(() => {
  //   console.log(nodes.length);
  // }, [handleSetUndoListWhenUpdateNodeParameterIsCalled]);

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
    
    // Use setTimeout to access current state
    setTimeout(() => {

      const currentNodes = nodesRef.current;
      const currentEdges = edgesRef.current;

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
      
      console.log(operationType, newType);

      // Update selected nodes
      setSelectedNodes((oldNodes: any[]) =>
        oldNodes.map(e => e.id === elementID ? {...e, 
          id: newNodeId,
          data: {...e.data, operationType: operationType, type: newType, label: newType, parameters : newParameters || {}}} : e)
      );
      
      // Update edges with new node ID
      const updatedEdges = currentEdges.map(edge => ({
        ...edge,
        source: edge.source === elementID ? newNodeId : edge.source,
        target: edge.target === elementID ? newNodeId : edge.target
      }));

      const updatedNodes = currentNodes.map(e => e.id === elementID ? {...e, 
        id: newNodeId,
        data: {
          ...e.data,
          operationType: operationType,
          type: newType, 
          label: newType, 
          parameters: newParameters || {},
          inputChannels: channelData.inputChannels,
          outputChannels: channelData.outputChannels,
          can_inherit_from_parent: canInherit
        }} : e);

      const propNodes = propagateChannelInheritance(
        newNodeId,
        channelData.outputChannels,
        updatedNodes,
        updatedEdges,
        defaultLayers,
        defaultTensorOps,
        defaultActivators,
        defaultInputs
      ); 

      // Update node with new ID and properties and combine with changes to other nodes due to propagation
      setNodes(propNodes);
      setEdges(updatedEdges);
      
      handleSetUndoList(updatedNodes || [], updatedEdges || []);
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
    let newNodes: any[] = nodesRef.current.filter((e) => e.id !== elementID); // Remove the node from nodes state
    let newEdges: any[] = edgesRef.current.filter((e) => !(e.source === elementID || e.target === elementID)); // remove edges from node
    // Remove the node from selected nodes
    setSelectedNodes(oldNodes =>
      oldNodes.filter((e) => e.id !== elementID)
    );

    setNodes(newNodes);
    setEdges(newEdges);

    handleSetUndoList(newNodes, newEdges);
  };

  // Custom hook to handle exporting the current canvas state
  const handleExport = useExport(
    nodes, 
    edges, 
    defaultLayers, 
    defaultTensorOps, 
    defaultActivators,
    (msg) => showSnackbar(msg, 'success'),
    (msg) => showSnackbar(msg, 'error')
  );

const handleSave = async () => { //gets screenshot of canvas then saves
    if (canvasRef.current === null) return;

    const dataURL = await toPng(canvasRef.current, { cacheBust: true, });
    const base64Image = dataURL.replace(/^data:image\/png;base64,/, ''); // png to base64 conversion

    save(nodes, edges, name, base64Image,
      (msg) => {
        showSnackbar(msg, 'success');
        // Update last saved state after successful save
        setLastSavedState({
          nodes: JSON.parse(JSON.stringify(nodes)),
          edges: JSON.parse(JSON.stringify(edges)),
          name: name
        });
        setHasUnsavedChanges(false);
      },
      (msg) => showSnackbar(msg, 'error')
    )
  };

  const save = useSave(); //ensures hook is at the top

  // Track changes to detect unsaved modifications
  useEffect(() => {
    if (!lastSavedState) {
      // No saved state yet, consider it as having changes if nodes/edges exist
      setHasUnsavedChanges(nodes.length > 0 || edges.length > 0 || name !== "Untitled");
      return;
    }

    // Compare current state with last saved state
    const hasChanges = 
      JSON.stringify(nodes) !== JSON.stringify(lastSavedState.nodes) ||
      JSON.stringify(edges) !== JSON.stringify(lastSavedState.edges) ||
      name !== lastSavedState.name;
    
    setHasUnsavedChanges(hasChanges);
  }, [nodes, edges, name, lastSavedState]);

  // Handle browser navigation/refresh warning
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (hasUnsavedChanges) {
        e.preventDefault();
        e.returnValue = ''; // Modern browsers require this
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [hasUnsavedChanges]);

  // Handle custom navigation within the app
  const handleNavigate = (url: string) => {
    if (hasUnsavedChanges) {
      setPendingNavigation(url);
      setShowUnsavedDialog(true);
    } else {
      router.push(url);
    }
  };

  const handleConfirmLeave = () => {
    if (pendingNavigation) {
      setHasUnsavedChanges(false); // Disable beforeunload warning
      router.push(pendingNavigation);
    }
    setShowUnsavedDialog(false);
    setPendingNavigation(null);
  };

  const handleCancelLeave = () => {
    setShowUnsavedDialog(false);
    setPendingNavigation(null);
  }; 

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

  // calls use parse using nodes and edges references 
  const handleUseParse = () => {
    if (nodesRef.current != undefined && edgesRef.current != undefined){
      useParse(nodesRef.current, edgesRef.current).then(res => setErrors(res));
    };
  };

  // calls use parse every 250ms
  useEffect(() => {
    const PARSE_INTERVAL = 50;
    const parseIntervalID = setInterval(handleUseParse, PARSE_INTERVAL);
  }, []);  

  const getSetters = useCallback(() => {
    return {
      updateNodeParameter     : updateNodeParameter,
      updateNodeType          : updateNodeType,
      updateNodeOperationType : updateNodeOperationType,
      deleteNode              : deleteNode,
      handleSetUndoListWhenUpdateNodeParameterIsCalled : handleSetUndoListWhenUpdateNodeParameterIsCalled
    }
  }, [updateNodeParameter, updateNodeType, updateNodeOperationType, deleteNode, handleSetUndoListWhenUpdateNodeParameterIsCalled]);

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
      <AppBarHeader open={open} setOpen={setOpen} doUndo={doUndo} doRedo={doRedo} name={name} setName={setName} hasUnsavedChanges={hasUnsavedChanges} onNavigate={handleNavigate}/>
      {/* Sidebar with menu and export functionality */}
      <Sidebar
        open={open}
        setOpen={setOpen}
        selectedMenu={selectedMenu}
        setSelectedMenu={setSelectedMenu}
        nodes={nodes}
        addNode={addNode}
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
        <ReactFlowProvider>
          <div ref={canvasRef}>
            <Canvas
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            OnEdgesDelete={OnEdgesDelete}
            onConnect={onConnect}
            onSelectionChange={onSelectionChange}
            setEdges={setEdges}
            handleExport={handleExport}
            handleSave={handleSave}
            errorMessages={errorMsgs}
            />
          </div>
        </ReactFlowProvider>
      </Main>
      <Snackbar
        open={snackbar.open}
        autoHideDuration={5000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert 
          onClose={handleCloseSnackbar} 
          severity={snackbar.severity}
          variant="filled"
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
      
      {/* Unsaved Changes Warning Dialog */}
      <UnsavedChangesDialog
        open={showUnsavedDialog}
        onStay={handleCancelLeave}
        onLeave={handleConfirmLeave}
      />
    </Box>
  );
}

// Wrap with Suspense to handle useSearchParams
export default function CanvasPage() {
  return (
    <Suspense fallback={<Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}><Typography>Loading...</Typography></Box>}>
      <CanvasPageContent />
    </Suspense>
  );
}
