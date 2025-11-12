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
    <Dialog onClose={handleClose} open={open}>
      <DialogTitle>Are you sure you want to delete this neural network?</DialogTitle>
      <Box sx={{display: "flex", flexWrap: "wrap", justifyContent:"space-between", p: 4}}>
        <Button onClick={handleClose}>No</Button>
        <Button onClick={handleConfirm}>Yes</Button>
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