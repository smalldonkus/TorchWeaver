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

interface Props {
  nodes: any[];
  setNodes: (val: any) => void;
  defaultActivators: any[];
}

export default function ActivatorsForm({ nodes, setNodes, defaultActivators }: Props) {
  // Use parameter handling hook
  const { 
    parameters, 
    hasValidationErrors, 
    handleParameterChange, 
    handleValidationChange, 
    updateParameters 
  } = useParameterHandling();

  const [chosenActivator, setChosenActivator] = useState(defaultActivators?.[0] || null);

  // Update parameters when chosen activator changes
  useEffect(() => {
    if (defaultActivators && defaultActivators.length > 0) {
      setChosenActivator(defaultActivators[0]);
      updateParameters(defaultActivators[0]?.parameters || {});
    }
  }, [defaultActivators, updateParameters]);

  if (!defaultActivators || defaultActivators.length === 0) {
    return <div>Loading activators...</div>;
  }

  function setActivator(type: string) {
    const newActivator = defaultActivators.find((a) => a.type === type);
    setChosenActivator(newActivator);
    updateParameters(newActivator?.parameters || {});
  }

  const addActivator = () => {
    if (hasValidationErrors) {
      alert("Please fix parameter errors before adding the activator.");
      return;
    }

    const newId = generateUniqueNodeId("activator", nodes);
    setNodes([
      ...nodes,
      {
        id: newId,
        position: { x: 300, y: 100 + nodes.length * 60 },
        data: {
          label: chosenActivator.type,
          operationType: "Activator",
          type: chosenActivator.type,
          parameters: parameters,
          outgoing_edges_count: 0
        },
      },
    ]);
    setChosenActivator(defaultActivators[0]);
    updateParameters(defaultActivators[0]?.parameters || {});
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="subtitle1" sx={{ mb: 2 }}>
        Add Activation Function
      </Typography>
      <TextField
        select
        label="Activation Function Type"
        value={chosenActivator?.type || ""}
        onChange={e => setActivator(e.target.value)}
        fullWidth
        size="small"
        sx={{ mb: 2 }}
      >
        {defaultActivators.map((a) => (
          <MenuItem key={a.type} value={a.type}>{a.type}</MenuItem>
        ))}
      </TextField>
      
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