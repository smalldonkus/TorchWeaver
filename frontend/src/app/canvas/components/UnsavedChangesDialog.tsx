"use client";

import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogTitle from "@mui/material/DialogTitle";
import Button from "@mui/material/Button";

interface Props {
  open: boolean;
  onStay: () => void;
  onLeave: () => void;
}

export default function UnsavedChangesDialog({ open, onStay, onLeave }: Props) {
  return (
    <Dialog
      open={open}
      onClose={onStay}
      aria-labelledby="unsaved-changes-dialog-title"
      aria-describedby="unsaved-changes-dialog-description"
      PaperProps={{
        sx: {
          borderRadius: '12px',
          padding: '8px',
          fontFamily: 'inherit',
        }
      }}
    >
      <DialogTitle 
        id="unsaved-changes-dialog-title"
        sx={{
          fontFamily: 'inherit',
          fontWeight: 600,
          fontSize: '1.4rem',
          color: '#FF7700',
        }}
      >
        Unsaved Changes
      </DialogTitle>
      <DialogContent>
        <DialogContentText 
          id="unsaved-changes-dialog-description"
          sx={{
            fontFamily: 'inherit',
            fontSize: '1rem',
            color: '#333',
          }}
        >
          You have unsaved changes. Are you sure you want to leave? Your changes will be lost.
        </DialogContentText>
      </DialogContent>
      <DialogActions sx={{ padding: '16px 24px', gap: '12px' }}>
        <Button 
          onClick={onStay} 
          variant="outlined"
          sx={{
            fontFamily: 'inherit',
            textTransform: 'none',
            fontWeight: 600,
            fontSize: '1rem',
            borderRadius: '8px',
            borderColor: '#FF7700',
            color: '#FF7700',
            '&:hover': {
              borderColor: '#f88f34',
              backgroundColor: 'rgba(255, 119, 0, 0.04)',
            }
          }}
        >
          Stay
        </Button>
        <Button 
          onClick={onLeave} 
          variant="contained"
          autoFocus
          sx={{
            fontFamily: 'inherit',
            textTransform: 'none',
            fontWeight: 600,
            fontSize: '1rem',
            borderRadius: '8px',
            backgroundColor: '#d32f2f',
            color: 'white',
            '&:hover': {
              backgroundColor: '#b71c1c',
            }
          }}
        >
          Leave Without Saving
        </Button>
      </DialogActions>
    </Dialog>
  );
}
