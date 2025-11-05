import { Box, Button, Grid, MenuItem, Popover, Stack, TextField, Typography} from "@mui/material";
import { useState, useEffect} from "react"
import ErrorIcon from '@mui/icons-material/Error';
import SettingsIcon from '@mui/icons-material/Settings';
import { useParameterHandling } from "../hooks/useParameterHandling";
import ParameterInputsTorchNode from "./ParamterInputsTorchNode";
import { Handle, Position } from "@xyflow/react";

// props contains all the nodes variables (label, id, data ... etc)
export default function TorchNode(props) {

  // SYSTEM CONSTANTS
  const DEBUG = true;
  // SYSTEM CONSTANTS

  const [anchorEP, setAnchorEP] = useState<HTMLButtonElement | null>(null);

  const [canEdit, setCanEdit] = useState<Boolean>(false) // two states, "view" | "edit"

  const [updateNodeParameter, setUpdateNodeParameter]         = useState<Function>(() => {});
  const [updateNodeType, setUpdateNodeType]                   = useState<Function>(() => {});
  const [updateNodeOperationType, setUpdateNodeOperationType] = useState<Function>(() => {});
  const [deleteNode, setDeleteNode]                           = useState<Function>(() => {});

  const [defaultLayers, setDefaultLayers]         = useState<any>(null);
  const [defaultTensorOps, setDefaultTensorOps]   = useState<any>(null);
  const [defaultActivators, setDefaultActivators] = useState<any>(null); // this seems quite bad (TN)
  const [defaults, setDefaults]                   = useState<any>(null);

  const isErrorPopoverOpen = Boolean(anchorEP);
  const idEP = isErrorPopoverOpen ? "error-popover" : undefined;

  const hasError = props.data.errors.length != 0

  const openErrorPopover = (event: React.MouseEvent<HTMLButtonElement>) => {
    setAnchorEP(event.currentTarget);
  };
  const closeErrorPopover = () => {
    setAnchorEP(null);
  };
  
  const toggleState = () => {
    setCanEdit(!canEdit);
  }

  // initialise setters for the node
  useEffect(() => {
    const setters = props.data.getSetters();
    setUpdateNodeParameter(     () => setters.updateNodeParameter); 
    setUpdateNodeType(          () => setters.updateNodeType);
    setUpdateNodeOperationType( () => setters.updateNodeOperationType);
    setDeleteNode(              () => setters.deleteNode);

    const localDefaults = props.data.getDefaults();
    setDefaultLayers(localDefaults.defaultLayers);
    setDefaultTensorOps(localDefaults.defaultTensorOps);
    setDefaultActivators(localDefaults.defaultActivators);
    setDefaults({
      Layer : defaultLayers,
      Activator: defaultActivators,
      TensorOp : defaultTensorOps
    })
  }, [canEdit]); // update only when canEdit is changed (reduces calls)
  
  /*
  Rohin's (and some what Toby's) EditLayerForm Code (3-11-2025) (TN)
  */
 const { 
      parameters, 
      hasValidationErrors, 
      handleParameterChange, 
      handleValidationChange, 
      updateParameters 
  } = useParameterHandling(); 
  
  // State for hierarchical selection
  const [selectedOperationType, setSelectedOperationType] = useState<string>("");
  const [selectedClass, setSelectedClass] = useState<string>("");
  const [selectedSpecificType, setSelectedSpecificType] = useState<string>("");
  
  // State for tracking pending changes
  const [hasPendingChanges, setHasPendingChanges] = useState(false);

  // Helper functions for getting available options
  const getAvailableClasses = (operationType: string): string[] => {
      let data;
      switch (operationType) {
          case "Layer":
              data = defaultLayers;
              break;
          case "TensorOp":
              data = defaultTensorOps;
              break;
          case "Activator":
              data = defaultActivators;
              break;
          default:
              return [];
      }
      return data?.data ? Object.keys(data.data) : [];
  };

  // Get available specific types based on operation type and class
  const getAvailableSpecificTypes = (operationType: string, className: string): string[] => {
      let data;
      switch (operationType) {
          case "Layer":
              data = defaultLayers;
              break;
          case "TensorOp":
              data = defaultTensorOps;
              break;
          case "Activator":
              data = defaultActivators;
              break;
          default:
              return [];
      }
      return data?.data?.[className] ? Object.keys(data.data[className]) : [];
  };


  useEffect(() => {
    if (props != null) {
        setSelectedOperationType(props.data.operationType || "");
        setSelectedSpecificType(props.data.type || "");
        updateParameters(props.data.parameters || {});
        setHasPendingChanges(false);
    }
  }, [props.id, canEdit]); // may need to be expanded

  // Initialize selected class based on current type
  useEffect(() => {
      if (selectedOperationType && selectedSpecificType) {
          // Find which class contains the current specific type
          const availableClasses = getAvailableClasses(selectedOperationType);
          for (const className of availableClasses) {
              const specificTypes = getAvailableSpecificTypes(selectedOperationType, className);
              if (specificTypes.includes(selectedSpecificType)) {
                  setSelectedClass(className);
                  break;
              }
          }
      }
  }, [selectedOperationType, selectedSpecificType, defaultLayers, defaultTensorOps, defaultActivators]);

  // Early return AFTER all hooks are declared
  if (!props) { // potential error
      return null;
  }

  const deleteNodeLocal = () => {
      if (deleteNode == undefined || deleteNode == null){
        return;
      }
      deleteNode(props.id);
  };

  // Handle operation type change
  const handleOperationTypeChange = (newOperationType: string) => {
      setSelectedOperationType(newOperationType);
      setSelectedClass("");
      setSelectedSpecificType("");
      setHasPendingChanges(true);
  };

  // Handle class change
  const handleClassChange = (newClass: string) => {
      setSelectedClass(newClass);
      setSelectedSpecificType("");
      setHasPendingChanges(true);
  };

  // Handle specific type change
  const handleSpecificTypeChange = (newSpecificType: string) => {

      if (!(selectedOperationType && selectedClass)) {
        return;
      };
      setSelectedSpecificType(newSpecificType);
      // set parameters to the default of that class,
      updateParameters(defaults[selectedOperationType].data[selectedClass][newSpecificType].parameters);
      setHasPendingChanges(true);
  };

  // Wrap the parameter change handler to track pending changes
  const handleParameterChangeWithPending = (parameterKey: string, value: any) => {
      handleParameterChange(parameterKey, value);
      setHasPendingChanges(true);
  };

  // Apply all pending changes
  const handleApplyEdit = () => {
      if (!props) return;
      if (
        updateNodeOperationType == undefined ||
        updateNodeType == undefined ||
        updateNodeParameter == undefined
      ) {
        console.log("Setters undefined");
        return;
      }

      // Apply operation type change if different
      if ((selectedOperationType && selectedOperationType !== props.data.operationType)
          &&
          (selectedSpecificType  && selectedSpecificType  !== props.data.type)){
        updateNodeOperationType(props.id, selectedOperationType, selectedSpecificType, parameters);
      }      
      // Apply specific type change if different
      else if (selectedSpecificType && selectedSpecificType !== props.data.type) {
        updateNodeType(props.id, selectedOperationType, selectedSpecificType, parameters);
      }
      else {
        // Apply parameter changes
        Object.entries(parameters).forEach(([key, value]) => {
            updateNodeParameter(props.id, key, value);
        });
      }
      // Reset pending changes
      setHasPendingChanges(false);
  };

  // styling
  const errorIconBorderColourMUI = hasError ?  "error" : "primary";
  const buttonSx = {minWidth:0, padding:"5px"};
  const gridSizes = {
    column: 36,
    item: 12
  }

  return (
    <div className="torch-node">
    <Handle type="target" position={Position.Top}/>
    <Handle type="source" position={Position.Bottom}/>
      <Box 
        sx={{
            display: "flex", 
            flexDirection:"column", 
            padding: "5px",
            gap: "10px",
            width: canEdit ? "600px" : "auto"
        }}>
          <Box 
            sx={{
              display: "flex", 
              flexDirection: "row",
              justifyContent: "flex-start",
              alignItems: "center",
              gap: "10px",
            }}>
              <Stack>
                <Typography sx={{}} variant="h5">{props.data.label}</Typography>
                {canEdit && DEBUG && <Typography sx={{}} variant="caption" color="#757575"> {props.id}</Typography>}
              </Stack>
              
              <Button className="nodrag" onClick={toggleState} variant="outlined" color="primary" sx={buttonSx}>
                  <SettingsIcon/>
              </Button>
              <Button className="nodrag" onClick={openErrorPopover} variant="outlined" color={errorIconBorderColourMUI} sx={buttonSx}>
                  <ErrorIcon 
                    color={hasError ? "error" : "primary" }
                  />
              </Button>
              <Popover
                id = {idEP}
                open = {isErrorPopoverOpen}
                onClose={closeErrorPopover}
                anchorEl={anchorEP}
                anchorOrigin={{
                  vertical: "center",
                  horizontal: "right"
                }}
                sx = {{
                  padding: "5px"
                }}
              >
                {hasError && props.data.errors.map((e, i) => (
                  <Typography key = {i} sx={{ p: 2 }} variant="h5">{e}</Typography>
                ))}
                {!hasError && 
                  <Typography sx={{ p: 2 }} variant="h5">No Errors To Display</Typography>
                }
              </Popover>
          </Box>
          {/* this box */}
          {(canEdit && (defaultLayers != null) && (defaultActivators != null) && (defaultTensorOps != null)) && (
            <Box sx={{flexGrow: 1}}>
              {/* <Grid container spacing={{ xs: 2, md: 3 }} columns={{ xs: 4, sm: 8, md: 12 }}> */}
              {/* TODO: work this out ^^  */ }
              {!(props.data.operationType === "Input" || props.data.operationType === "Output") && <Grid container spacing={2} columns={gridSizes.column}>
                <Grid display='flex' justifyContent='center' alignItems='center' size={gridSizes.item}>
                  <TextField
                      select
                      label="Operation Type"
                      value={selectedOperationType}
                      onChange={(e) => handleOperationTypeChange(e.target.value)}
                      fullWidth
                      size="small"
                      sx={{ mb: 2 }}
                      className="nodrag"
                  >   
                    <MenuItem key={"Layer"} value={"Layer"}>{"Layer"}</MenuItem>
                    <MenuItem key={"TensorOp"} value={"TensorOp"}>{"Tensor Operation"}</MenuItem>
                    <MenuItem key={"Activator"} value={"Activator"}>{"Activator"}</MenuItem>
                  </TextField>
                </Grid>
                <Grid display='flex' justifyContent='center' alignItems='center' size={gridSizes.item}>
                  {selectedOperationType && (
                    <TextField
                        select
                        label="Class"
                        value={selectedClass}
                        onChange={(e) => handleClassChange(e.target.value)}
                        fullWidth
                        size="small"
                        sx={{ mb: 2 }}
                        disabled={!selectedOperationType}
                        className="nodrag"
                    >   
                        {getAvailableClasses(selectedOperationType).map((className) => (
                            <MenuItem key={className} value={className}>{className}</MenuItem>
                        ))}
                    </TextField>
                )}
                </Grid>
                <Grid display='flex' justifyContent='center' alignItems='center' size={gridSizes.item}>
                  {selectedClass && (
                    <TextField
                        select
                        label="Specific Type"
                        value={selectedSpecificType}
                        onChange={(e) => handleSpecificTypeChange(e.target.value)}
                        fullWidth
                        size="small"
                        sx={{ mb: 2 }}
                        disabled={!selectedClass}
                        className="nodrag" // this is stupidest shit i've ever had to do (TN)
                    >   
                        {getAvailableSpecificTypes(selectedOperationType, selectedClass).map((specificType) => (
                            <MenuItem key={specificType} value={specificType}>{specificType}</MenuItem>
                        ))}
                    </TextField>
                )}
                </Grid>
              </Grid>}
                {selectedSpecificType && parameters && (
                    <ParameterInputsTorchNode
                        operationType={selectedOperationType as "Layer" | "TensorOp" | "Activator"}
                        nodeType={selectedSpecificType}
                        parameters={parameters}
                        onParameterChange={handleParameterChangeWithPending}
                        onValidationChange={handleValidationChange}
                        gridSizes={gridSizes}
                    />
                )}
              <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
                <Button 
                    variant="contained" 
                    fullWidth 
                    onClick={handleApplyEdit}
                    // requires that a operationType and specificType have been chosen
                    disabled={(!hasPendingChanges || hasValidationErrors) || selectedOperationType==="" || selectedSpecificType===""}
                    sx={{ backgroundColor: 'primary.main' }}
                >
                    Apply Edit
                </Button>
                <Button 
                    variant="contained" 
                    fullWidth 
                    style={{backgroundColor: "red"}} 
                    onClick={deleteNodeLocal}
                >
                    Delete
                </Button>
            </Box>
            </Box>
          )}
      </Box>
    </div>
  );
}