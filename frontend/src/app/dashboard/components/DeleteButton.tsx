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
}

function Popup(props: PopupProps) {
  const { onClose, open } = props;

  const handleClose = () => { //Close popup when user clicks off window
    onClose();
  }; 

  const handleListItemClick = () => { //Close popup when user clicks either button
    onClose();
  };

  return ( //Popup dialogue
    <Dialog onClose={handleClose} open={open}> 
      <DialogTitle>Are you sure you want to delete this neural network?</DialogTitle>
      <Box sx={{display: "flex", flexWrap: "wrap", justifyContent:"space-between", p: 4}}>
        <Button onClick={() => handleListItemClick()}>No</Button>
        <Button onClick={() => handleListItemClick()}>Yes</Button>
      </Box>
    </Dialog>
  );
}

export default function DeleteButton() {
  const [open, setOpen] = React.useState(false);

  const handleClickOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  return (
    <>
        <IconButton onClick={handleClickOpen}>
            <DeleteIcon />
            </IconButton>
            <Popup
                open={open}
                onClose={handleClose}
            />
    </>
  );
}