import { useState, useEffect } from "react";
import Box from "@mui/material/Box";
import TextField from "@mui/material/TextField";
import FormHelperText from "@mui/material/FormHelperText";
import Chip from "@mui/material/Chip";
import { validateParameter, getParameterHelperText } from "../utils/parameterValidation";
import { useNodeDefinitions, NodeType } from "../hooks/useNodeDefinitions";
import { useVariablesInfo } from "../hooks/useVariablesInfo";
import { Grid, Popover, Typography } from "@mui/material";

interface ParameterInputsProps {
  operationType: "Layer" | "TensorOp" | "Activator" | "Input";
  nodeType: string;
  parameters: Record<string, any>;
  onParameterChange: (parameterKey: string, value: any) => void;
  onValidationChange?: (hasErrors: boolean) => void;
  gridSizes: any;
  nodeDefinition?: any; // Add node definition to access ChannelLinks
}

/*
    Different verions from ParameterInputs to:
    A. return a grid box only
    B. Only display helper info on hover
*/

export default function ParameterInputs({
  operationType,
  nodeType,
  parameters,
  onParameterChange,
  onValidationChange,
  gridSizes,
  nodeDefinition
}: ParameterInputsProps) {
  // Get the correct backend endpoint name for the operation type
  const getBackendNodeType = (operationType: string): NodeType => {
    switch (operationType) {
      case "Layer":
        return "layers";
      case "TensorOp":
        return "tensorops";
      case "Activator":
        return "activators";
      case "Input":
        return "inputs";
      default:
        return "layers";
    }
  };

  // Fetch parameter format information from backend
  const { getParameterFormat } = useNodeDefinitions(getBackendNodeType(operationType));
  const { variablesInfo } = useVariablesInfo();
  
  // State for parameter validation errors
  const [parameterErrors, setParameterErrors] = useState<{ [key: string]: string }>({});


  // Notify parent component when validation state changes
  useEffect(() => {
    const hasErrors = Object.keys(parameterErrors).length > 0;
    onValidationChange?.(hasErrors);
  }, [parameterErrors, onValidationChange]);

  // Clear errors when node type changes
  useEffect(() => {
    setParameterErrors({});
  }, [nodeType]);

  // Update a parameter value with validation
  const updateParam = (parameterKey: string, parameterValue: string) => {
    // Get expected types for this parameter
    const expectedTypes = getParameterFormat(nodeType, parameterKey);
    
    // Validate the parameter
    const validation = validateParameter(parameterValue, expectedTypes);
    
    // Update parameter errors
    if (validation.isValid) {
      setParameterErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[parameterKey];
        return newErrors;
      });
    } else {
      setParameterErrors(prev => ({
        ...prev,
        [parameterKey]: validation.error || "Invalid value"
      }));
    }

    // Notify parent with the converted value (if valid) or original value (if not)
    onParameterChange(parameterKey, validation.isValid ? validation.convertedValue : parameterValue);
  };

  // Helper function to determine if a parameter should be disabled
  const isParameterDisabled = (parameterKey: string): boolean => {
    // Only disable if inherit_from_parent is true
    if (!parameters.inherit_from_parent) {
      return false;
    }

    // Check if this parameter is linked to inputParam in ChannelLinks
    if (nodeDefinition?.parseCheck?.ChannelLinks) {
      return nodeDefinition.parseCheck.ChannelLinks.some((link: any) => 
        link.inputParam === parameterKey
      );
    }

    return false;
  };

  if (!parameters) {
    return null;
  }



  // Helper function to get description for a single type
  const getTypeDescription = (type: string): string => {
    const typeInfo = variablesInfo.find(info => info.type === type);
    return typeInfo ? typeInfo.input_description : `Enter ${type} value`;
  };

  // State to track which chip popover is open
  const [chipPopoverAnchor, setChipPopoverAnchor] = useState<HTMLElement | null>(null);
  const [chipPopoverType, setChipPopoverType] = useState<string | null>(null);
  const isChipPopoverOpen = Boolean(chipPopoverAnchor);

  return (
    <Grid container spacing={2} columns={gridSizes.column}>
      {Object.keys(parameters).map((parameterKey, i) => {
        const expectedTypes = getParameterFormat(nodeType, parameterKey);
        const hasError = parameterErrors[parameterKey];
        const helperText = hasError || getParameterHelperText(expectedTypes, variablesInfo);
        const isDisabled = isParameterDisabled(parameterKey);
        
        return (
            <Grid key={i} size={gridSizes.item}>
                <Box key={i} sx={{ mb: 2 }}>
                    <TextField
                    label={parameterKey}
                    value={parameters[parameterKey]}
                    onChange={(e) => updateParam(parameterKey, e.target.value)}
                    error={!!hasError}
                    disabled={isDisabled}
                    fullWidth
                    size="small"
                    className="nodrag"
                    helperText={isDisabled ? "Parameter inherited from parent node" : undefined}
                    />
                    
                    {/* Show expected types as chips */}
                    {expectedTypes.length > 0 && !isDisabled && (
                    <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mt: 0.5 }}>
                        {expectedTypes.map((type, typeIndex) => (
                          <div key={type}>
                              <Chip
                                  label={type}
                                  size="small"
                                  variant="outlined"
                                  onClick={(e) => {
                                    setChipPopoverAnchor(e.currentTarget);
                                    setChipPopoverType(type);
                                  }}
                                  sx={{ fontSize: "0.7rem", height: "20px" }}
                                  className="nodrag"
                              />
                          </div>
                        ))}
                        {/* Single Popover for all chips, shows info for selected type */}
                        <Popover
                          open={isChipPopoverOpen}
                          onClose={() => {
                            setChipPopoverAnchor(null);
                            setChipPopoverType(null);
                          }}
                          anchorEl={chipPopoverAnchor}
                          anchorOrigin={{
                              vertical: "center",
                              horizontal: "right"
                          }}
                          sx={{
                            padding: "12px",
                            maxWidth: "1500px"
                          }}
                        >
                          <Typography sx={{ p: 1 }} variant="subtitle2">
                            {chipPopoverType ? getTypeDescription(chipPopoverType) : ""}
                          </Typography>
                        </Popover>
                    </Box>
                    )}
                    
                    {/* Helper text */}
                    {/* <FormHelperText error={!!hasError}>
                    {helperText}
                    </FormHelperText> */}
                    {/* Will be added back in, in a different way */}
                </Box>
            </Grid>
        );
      })}
    </Grid>
  );
}