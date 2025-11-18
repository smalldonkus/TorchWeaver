"use client";

// Import React hooks and MUI components
import { useState, useEffect } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import MenuItem from "@mui/material/MenuItem";
import Button from "@mui/material/Button";
import { generateUniqueNodeId } from "../utils/idGenerator";
import ParameterInputs from "./ParameterInputs";
import { useParameterHandling } from "../hooks/useParameterHandling";
import { createNode } from "./TorchNodeCreator";

// Define the props that this component expects
interface Props {
  nodes: any[]; // List of current nodes
  addNode: (val: any) => void; // Function to update nodes
  defaultInputs: any; // Input data with global classes structure
  getSetters: () => any; // for TorchNode functionality
  getDefaults: () => any; // for editing within a node
}

// Main component for the Input Form
export default function InputForm({ nodes, addNode, defaultInputs, getSetters, getDefaults}: Props) {
  // All hooks must be called before any conditional returns!
  
  // Use parameter handling hook
  const { 
    parameters, 
    hasValidationErrors, 
    handleParameterChange, 
    handleValidationChange, 
    updateParameters 
  } = useParameterHandling();
  
  // State for the currently selected global class and input type
  const [selectedClass, setSelectedClass] = useState<string>("");
  const [selectedInputType, setSelectedInputType] = useState<string>("");
  const [chosenDefault, setChosenDefault] = useState<any>(null);

  // Extract global classes from the new structure
  const globalClasses = defaultInputs?.data ? Object.keys(defaultInputs.data) : [];

  // When defaultInputs changes, set initial selections
  useEffect(() => {
    if (defaultInputs?.data && globalClasses.length > 0) {
      setSelectedClass(globalClasses[0]);
      const firstClassInputs = defaultInputs.data[globalClasses[0]];
      const firstInputType = Object.keys(firstClassInputs)[0];
      setSelectedInputType(firstInputType);
      setChosenDefault({
        class: globalClasses[0],
        type: firstInputType,
        ...firstClassInputs[firstInputType]
      });
      updateParameters(firstClassInputs[firstInputType]?.parameters || {});
    }
  }, [defaultInputs, updateParameters]);

  // If input types haven't loaded yet, show a loading message
  if (!defaultInputs || !defaultInputs.data || Object.keys(defaultInputs.data).length === 0) {
    return <div>Loading input types...</div>;
  }

  // Handle class selection change
  function handleClassChange(className: string) {
    setSelectedClass(className);
    const classInputs = defaultInputs.data[className];
    const firstInputType = Object.keys(classInputs)[0];
    setSelectedInputType(firstInputType);
    setChosenDefault({
      class: className,
      type: firstInputType,
      ...classInputs[firstInputType]
    });
    updateParameters(classInputs[firstInputType]?.parameters || {});
  }

  // Handle input type selection change within a class
  function handleInputTypeChange(inputType: string) {
    setSelectedInputType(inputType);
    const inputData = defaultInputs.data[selectedClass][inputType];
    setChosenDefault({
      class: selectedClass,
      type: inputType,
      ...inputData
    });
    updateParameters(inputData?.parameters || {});
  }

  // Add a new input node to the list
  const addInput = () => {
    // Check if there are any validation errors
    if (hasValidationErrors) {
      alert("Please fix parameter errors before adding the input.");
      return;
    }

    const newId = generateUniqueNodeId("input", nodes);
    const newNode = createNode(
      newId,
      nodes.length, // posModifier
      selectedInputType, // label
      "Input", // operation type
      selectedInputType, // type
      parameters, // parameters from the form
      getSetters,
      getDefaults,
      chosenDefault
    );
    
    addNode([...nodes, newNode]);
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="subtitle1" sx={{ mb: 2 }}>
        Add Input
      </Typography>
      
      {/* Global Class Selection */}
      <TextField
        select
        label="Input Category"
        value={selectedClass}
        onChange={e => handleClassChange(e.target.value)}
        fullWidth
        size="small"
        sx={{ mb: 2 }}
      >
        {globalClasses.map((className) => (
          <MenuItem key={className} value={className}>
            {className}
          </MenuItem>
        ))}
      </TextField>

      {/* Input Type Selection within the selected class */}
      {selectedClass && (
        <TextField
          select
          label="Input Type"
          value={selectedInputType}
          onChange={e => handleInputTypeChange(e.target.value)}
          fullWidth
          size="small"
          sx={{ mb: 2 }}
        >
          {Object.keys(defaultInputs.data[selectedClass]).map((inputType) => (
            <MenuItem key={inputType} value={inputType}>
              {inputType}
            </MenuItem>
          ))}
        </TextField>
      )}

      {/* Parameter Inputs */}
      {chosenDefault && (
        <ParameterInputs
          operationType="Input"
          nodeType={chosenDefault.type}
          parameters={parameters}
          onParameterChange={handleParameterChange}
          onValidationChange={handleValidationChange}
          nodeDefinition={chosenDefault}
        />
      )}

      {/* Add Input Button */}
      <Button 
          variant="contained" 
          fullWidth 
          onClick={addInput}
          sx={{ 
              backgroundColor: '#202A44',
              borderRadius: '8px',
              textTransform: 'none',
              fontWeight: 600,
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
              '&:hover': {
                  backgroundColor: '#2d3a5e',
                  boxShadow: '0 4px 8px rgba(0,0,0,0.2)'
              }
          }}
      >
        Add Input
      </Button>
    </Box>
  );
}