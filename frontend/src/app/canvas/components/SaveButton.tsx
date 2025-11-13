"use client";
import React from 'react';
import IconButton from "@mui/material/IconButton";
import SaveIcon from "@mui/icons-material/Save";

interface SaveButtonProps {
  handleSave: () => void;
}

export default function SaveButton({ handleSave }: SaveButtonProps) {
  return (
    <IconButton
      onClick={handleSave}
      sx={{
        position: "absolute",
        top: 145,
        right: 16,
        backgroundColor: "#e2c338ff",
        color: "white",
        width: 48,
        height: 48,
        zIndex: 1000,
        boxShadow: 3,
        "&:hover": {
          backgroundColor: "#26489cff",
        },
      }}
    >
      <SaveIcon />
    </IconButton>
  );
}
