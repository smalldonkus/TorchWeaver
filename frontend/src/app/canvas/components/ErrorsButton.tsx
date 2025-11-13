"use client";
import React, { useState } from 'react';
import IconButton from "@mui/material/IconButton";
import ErrorIcon from "@mui/icons-material/Error";
import Popper from "@mui/material/Popper";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";
import ClickAwayListener from "@mui/material/ClickAwayListener";
import Divider from "@mui/material/Divider";

interface ErrorsButtonProps {
  errorMessages: string[];
}

export default function ErrorsButton({ errorMessages }: ErrorsButtonProps) {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const open = Boolean(anchorEl);

  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(anchorEl ? null : event.currentTarget);
  };

  const handleClickAway = () => {
    setAnchorEl(null);
  };

  const hasErrors = errorMessages && errorMessages.length > 0;

  return (
    <>
      <IconButton
        onClick={handleClick}
        sx={{
          position: "absolute",
          top: 25,
          right: 16,
          backgroundColor: hasErrors ? "#d32f2f" : "#4caf50",
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
      <Popper
        open={open}
        anchorEl={anchorEl}
        placement="left-start"
        sx={{ zIndex: 1001 }}
      >
        <ClickAwayListener onClickAway={handleClickAway}>
          <Paper
            elevation={6}
            sx={{
              p: 2,
              maxWidth: 400,
              maxHeight: 400,
              overflow: "auto",
              backgroundColor: "#fff",
              border: hasErrors ? "2px solid #d32f2f" : "2px solid #4caf50",
              borderRadius: 3,
            }}
          >
            <Typography
              variant="h6"
              sx={{
                fontWeight: 700,
                fontSize: '1.1rem',
                color: hasErrors ? "#d32f2f" : "#4caf50",
                mb: 1,
                fontFamily: 'inherit',
              }}
            >
              {hasErrors ? `Errors` : "No Errors"}
            </Typography>
            <Divider sx={{ mb: 2 }} />
            {hasErrors ? (
              <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
                {errorMessages.map((error, index) => (
                  <Box
                    key={index}
                    sx={{
                      p: 1.5,
                      backgroundColor: "#ffebee",
                      borderRadius: 1,
                      borderLeft: "4px solid #d32f2f",
                    }}
                  >
                    <Typography 
                      variant="body2" 
                      sx={{ 
                        color: "#333",
                        fontFamily: 'inherit',
                        lineHeight: 1.6,
                      }}
                    >
                      <strong style={{ fontWeight: 600 }}>Error {index + 1}:</strong> {error}
                    </Typography>
                  </Box>
                ))}
              </Box>
            ) : (
              <Typography 
                variant="body2" 
                sx={{ 
                  color: "#666",
                  fontFamily: 'inherit',
                  lineHeight: 1.6,
                }}
              >
                All validations passed successfully!
              </Typography>
            )}
          </Paper>
        </ClickAwayListener>
      </Popper>
    </>
  );
}
