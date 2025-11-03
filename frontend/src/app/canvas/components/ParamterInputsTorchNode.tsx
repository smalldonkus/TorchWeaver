import { useState, useEffect } from "react";
import Box from "@mui/material/Box";
import TextField from "@mui/material/TextField";
import FormHelperText from "@mui/material/FormHelperText";
import Chip from "@mui/material/Chip";
import { validateParameter, getParameterHelperText } from "../utils/parameterValidation";
import { useNodeDefinitions, NodeType } from "../hooks/useNodeDefinitions";
import { useVariablesInfo } from "../hooks/useVariablesInfo";
import { Grid } from "@mui/material";

interface ParameterInputsProps {
  operationType: "Layer" | "TensorOp" | "Activator";
  nodeType: string;
  parameters: Record<string, any>;
  onParameterChange: (parameterKey: string, value: any) => void;
  onValidationChange?: (hasErrors: boolean) => void;
  gridSizes: any;
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
  gridSizes
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

  if (!parameters) {
    return null;
  }

  return (
    <Grid container spacing={2} columns={gridSizes.column}>
      {Object.keys(parameters).map((parameterKey, i) => {
        const expectedTypes = getParameterFormat(nodeType, parameterKey);
        const hasError = parameterErrors[parameterKey];
        const helperText = hasError || getParameterHelperText(expectedTypes, variablesInfo);
        
        return (
            <Grid key={i} size={gridSizes.item}>
                <Box key={i} sx={{ mb: 2 }}>
                    <TextField
                    label={parameterKey}
                    value={parameters[parameterKey]}
                    onChange={(e) => updateParam(parameterKey, e.target.value)}
                    error={!!hasError}
                    fullWidth
                    size="small"
                    />
                    
                    {/* Show expected types as chips */}
                    {expectedTypes.length > 0 && (
                    <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mt: 0.5 }}>
                        {expectedTypes.map((type) => (
                        <Chip
                            key={type}
                            label={type}
                            size="small"
                            variant="outlined"
                            sx={{ fontSize: "0.7rem", height: "20px" }}
                        />
                        ))}
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