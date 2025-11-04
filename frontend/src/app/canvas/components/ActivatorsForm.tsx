"use client";

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

interface Props {
  nodes: any[];
  setNodes: (val: any) => void;
  defaultActivators: any; // Changed from any[] to any to handle new structure
  // for TorchNode functionality, allows it to update itself (TN)
  getSetters: () => any;
  getDefaults: () => any; // for editing within a node (TN)
}

export default function ActivatorsForm({ nodes, setNodes, defaultActivators, getSetters, getDefaults}: Props) {
  // Use parameter handling hook
  const { 
    parameters, 
    hasValidationErrors, 
    handleParameterChange, 
    handleValidationChange, 
    updateParameters 
  } = useParameterHandling();

  const [selectedClass, setSelectedClass] = useState<string>("");
  const [selectedActivatorType, setSelectedActivatorType] = useState<string>("");
  const [chosenActivator, setChosenActivator] = useState<any>(null);

  // Extract global classes from the new structure
  const globalClasses = defaultActivators?.data ? Object.keys(defaultActivators.data) : [];

  // Initialize selections when data loads
  useEffect(() => {
    if (globalClasses.length > 0) {
      const firstClass = globalClasses[0];
      const firstClassActivators = defaultActivators.data[firstClass];
      const firstActivatorType = Object.keys(firstClassActivators)[0];
      
      setSelectedClass(firstClass);
      setSelectedActivatorType(firstActivatorType);
      setChosenActivator({
        class: firstClass,
        type: firstActivatorType,
        ...firstClassActivators[firstActivatorType]
      });
      updateParameters(firstClassActivators[firstActivatorType]?.parameters || {});
    }
  }, [defaultActivators, updateParameters]);

  if (!defaultActivators || !defaultActivators.data || Object.keys(defaultActivators.data).length === 0) {
    return <div>Loading activators...</div>;
  }

  function handleClassChange(className: string) {
    setSelectedClass(className);
    const classActivators = defaultActivators.data[className];
    const firstActivatorType = Object.keys(classActivators)[0];
    setSelectedActivatorType(firstActivatorType);
    const newActivator = {
      class: className,
      type: firstActivatorType,
      ...classActivators[firstActivatorType]
    };
    setChosenActivator(newActivator);
    updateParameters(newActivator?.parameters || {});
  }

  function handleActivatorTypeChange(activatorType: string) {
    setSelectedActivatorType(activatorType);
    const activatorData = defaultActivators.data[selectedClass][activatorType];
    const newActivator = {
      class: selectedClass,
      type: activatorType,
      ...activatorData
    };
    setChosenActivator(newActivator);
    updateParameters(newActivator?.parameters || {});
  }

  const addActivator = () => {
    if (hasValidationErrors) {
      alert("Please fix parameter errors before adding the activator.");
      return;
    }

    const newId = generateUniqueNodeId("activator", nodes);
    const newNode = createNode(
        newId,
        nodes.length, // posModifier
        chosenActivator.type, // label
        "Activator", // operation type
        chosenActivator.type, // type
        parameters,
        getSetters,
        getDefaults
    );
    setNodes([
      ...nodes,
      newNode
    ]);
    
    // Reset to first selection after adding
    if (globalClasses.length > 0) {
      const firstClass = globalClasses[0];
      const firstClassActivators = defaultActivators.data[firstClass];
      const firstActivatorType = Object.keys(firstClassActivators)[0];
      
      setSelectedClass(firstClass);
      setSelectedActivatorType(firstActivatorType);
      setChosenActivator({
        class: firstClass,
        type: firstActivatorType,
        ...firstClassActivators[firstActivatorType]
      });
      updateParameters(firstClassActivators[firstActivatorType]?.parameters || {});
    }
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="subtitle1" sx={{ mb: 2 }}>
        Add Activation Function
      </Typography>
      
      {/* Dropdown to select activator class */}
      <TextField
        select
        label="Activation Class"
        value={selectedClass}
        onChange={e => handleClassChange(e.target.value)}
        fullWidth
        size="small"
        sx={{ mb: 2 }}
      >
        {globalClasses.map((className) => (
          <MenuItem key={className} value={className}>{className}</MenuItem>
        ))}
      </TextField>

      {/* Dropdown to select specific activation function type within the class */}
      {selectedClass && (
        <TextField
          select
          label="Activation Function Type"
          value={selectedActivatorType}
          onChange={e => handleActivatorTypeChange(e.target.value)}
          fullWidth
          size="small"
          sx={{ mb: 2 }}
        >
          {Object.keys(defaultActivators.data[selectedClass]).map((activatorType) => (
            <MenuItem key={activatorType} value={activatorType}>{activatorType}</MenuItem>
          ))}
        </TextField>
      )}
      
      {chosenActivator && (
        <ParameterInputs
          operationType="Activator"
          nodeType={chosenActivator.type}
          parameters={parameters}
          onParameterChange={handleParameterChange}
          onValidationChange={handleValidationChange}
        />
      )}
      
      <Button variant="contained" fullWidth onClick={addActivator}>
        Add Activation Function
      </Button>
    </Box>
  );
}