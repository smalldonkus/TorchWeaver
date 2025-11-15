"use client";
import React from 'react';
import IconButton from "@mui/material/IconButton";
import SaveIcon from "@mui/icons-material/Save";
import { useUser } from "@auth0/nextjs-auth0/client";

interface SaveButtonProps {
  handleSave: () => void;
}

export default function SaveButton({ handleSave }: SaveButtonProps) {
  const { user } = useUser();
  const isDisabled = !user;

  return (
    <IconButton
      onClick={handleSave}
      disabled={isDisabled}
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
        "&.Mui-disabled": {
          backgroundColor: "#cccccc",
          color: "#888888",
        },
      }}
    >
      <SaveIcon />
    </IconButton>
  );
}
