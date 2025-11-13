"use client";
import React from 'react';
import IconButton from "@mui/material/IconButton";
import DownloadIcon from "@mui/icons-material/Download";

interface ExportButtonProps {
  handleExport: () => void;
}

export default function ExportButton({ handleExport }: ExportButtonProps) {
  return (
    <IconButton
      onClick={handleExport}
      sx={{
        position: "absolute",
        top: 85,
        right: 16,
        backgroundColor: "#f17d06",
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
      <DownloadIcon />
    </IconButton>
  );
}
