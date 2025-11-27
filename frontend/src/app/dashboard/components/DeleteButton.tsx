"use client";

import { Box, IconButton } from "@mui/material";
import DeleteIcon from '@mui/icons-material/Delete';
import * as React from 'react';
import DialogTitle from '@mui/material/DialogTitle';
import Dialog from '@mui/material/Dialog';
import Button from '@mui/material/Button';

export interface PopupProps {
  open: boolean;
  onClose: () => void;
  onConfirm: () => void;
}

function Popup(props: PopupProps) {
  const { onClose, onConfirm, open } = props;

  const handleClose = () => {
    onClose();
  };

  const handleConfirm = () => {
    onConfirm();
    onClose();
  };

  return (
    <Dialog 
      onClose={handleClose} 
      open={open}
      PaperProps={{
        sx: {
          borderRadius: '12px',
          padding: '8px',
          fontFamily: 'inherit',
        }
      }}
    >
      <DialogTitle
        sx={{
          fontFamily: 'inherit',
          fontWeight: 600,
          fontSize: '1.4rem',
          color: '#FF7700',
        }}
      >
        Delete Network
      </DialogTitle>
      <Box sx={{px: 3, pb: 2}}>
        <Box 
          sx={{
            fontFamily: 'inherit',
            fontSize: '1rem',
            color: '#333',
            mb: 3,
          }}
        >
          Are you sure you want to delete this network? This deletion cannot be undone.
        </Box>
        <Box sx={{display: "flex", gap: 2, justifyContent: "flex-end"}}>
          <Button 
            onClick={handleClose}
            sx={{
              fontFamily: 'inherit',
              fontWeight: 575,
              textTransform: 'none',
              borderRadius: '7px',
              padding: '7px 15px',
              color: '#666',
              '&:hover': {
                backgroundColor: '#f0f0f0',
              },
            }}
          >
            Cancel
          </Button>
          <Button 
            onClick={handleConfirm}
            variant="contained"
            sx={{
              fontFamily: 'inherit',
              fontWeight: 575,
              textTransform: 'none',
              borderRadius: '7px',
              padding: '7px 15px',
              backgroundColor: '#d32f2f',
              '&:hover': {
                backgroundColor: "#26489cff",
              },
            }}
          >
            Delete
          </Button>
        </Box>
      </Box>
    </Dialog>
  );
}

interface DeleteButtonProps {
  onClick?: () => void;
}

export default function DeleteButton({ onClick }: DeleteButtonProps) {
  const [open, setOpen] = React.useState(false);

  const handleClickOpen = (e: React.MouseEvent) => {
    e.stopPropagation(); // prevent parent card click
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  const handleConfirm = () => {
    if (onClick) onClick();
  };

  return (
    <>
      <IconButton onClick={handleClickOpen}>
        <DeleteIcon />
      </IconButton>
      <Popup open={open} onClose={handleClose} onConfirm={handleConfirm} />
    </>
  );
}