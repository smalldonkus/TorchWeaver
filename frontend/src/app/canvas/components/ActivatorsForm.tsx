"use client";

import { useState, useEffect } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import MenuItem from "@mui/material/MenuItem";
import Button from "@mui/material/Button";
import { generateUniqueNodeId } from "../utils/idGenerator";

interface Props {
  nodes: any[];
  setNodes: (val: any) => void;
  defaultActivators: any[];
}

export default function ActivatorsForm({ nodes, setNodes, defaultActivators }: Props) {
  if (!defaultActivators || defaultActivators.length === 0) {
    return <div>Loading activators...</div>;
  }

  const [chosenActivator, setChosenActivator] = useState(defaultActivators[0]);

  useEffect(() => {
    setChosenActivator(defaultActivators[0]);
  }, [defaultActivators]);

  function setActivator(type: string) {
    setChosenActivator(defaultActivators.find((a) => a.type === type));
  }

  const updateParam = (paramKey: string, paramValue: string) => {
    setChosenActivator(prev => ({
      ...prev,
      parameters: { ...prev.parameters, [paramKey]: paramValue }
    }));
  };

  const addActivator = () => {
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
          parameters: chosenActivator.parameters
        },
      },
    ]);
    setChosenActivator(defaultActivators[0]);
    setChosenActivator(defaultActivators[0]);
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="subtitle1" sx={{ mb: 2 }}>
        Add Activator
      </Typography>
      <TextField
        select
        label="Activator Type"
        value={chosenActivator?.type}
        onChange={e => setActivator(e.target.value)}
        fullWidth
        size="small"
        sx={{ mb: 2 }}
      >
        {defaultActivators.map((a) => (
          <MenuItem key={a.type} value={a.type}>{a.type}</MenuItem>
        ))}
      </TextField>
      {chosenActivator && chosenActivator.parameters &&
        Object.keys(chosenActivator.parameters).map((paramKey, i) => (
          <TextField
            key={i}
            label={paramKey}
            value={chosenActivator.parameters[paramKey]}
            onChange={e => updateParam(paramKey, e.target.value)}
            type={typeof chosenActivator.parameters[paramKey] === "number" ? "number" : "text"}
            fullWidth
            size="small"
            sx={{ mb: 2 }}
          />
        ))
      }
      <Button variant="contained" fullWidth onClick={addActivator}>
        Add Activator
      </Button>
    </Box>
  );
}