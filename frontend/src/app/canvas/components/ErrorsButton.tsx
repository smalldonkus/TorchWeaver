"use client";
import React from 'react';
import IconButton from "@mui/material/IconButton";
import ErrorIcon from "@mui/icons-material/Error";

interface ErrorsButtonProps {
  openErrorBox?: boolean;
  setOpenErrorBox: (val: boolean) => void;
}

export default function ErrorsButton({ openErrorBox, setOpenErrorBox }: ErrorsButtonProps) {
  return (
    <IconButton
      onClick={() => setOpenErrorBox(!openErrorBox)}
      sx={{
        position: "absolute",
        top: 25,
        right: 16,
        backgroundColor: "#d32f2f",
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
      <ErrorIcon />
    </IconButton>
  );
}
