"use client";
import React from 'react';
import IconButton from "@mui/material/IconButton";
import DownloadIcon from "@mui/icons-material/Download";
import { useUser } from "@auth0/nextjs-auth0/client";

interface ExportButtonProps {
  handleExport: () => void;
}

export default function ExportButton({ handleExport }: ExportButtonProps) {
  const { user } = useUser();
  const isDisabled = !user;

  return (
    <IconButton
      onClick={handleExport}
      disabled={isDisabled}
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
        "&.Mui-disabled": {
          backgroundColor: "#cccccc",
          color: "#888888",
        },
      }}
    >
      <DownloadIcon />
    </IconButton>
  );
}
