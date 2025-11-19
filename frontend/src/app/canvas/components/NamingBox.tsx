import React from 'react';
import Input from '@mui/material/Input';

export default function NamingBox({ value, onChange }) {
  return (
    <Input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      disableUnderline
      sx={{
        color: 'white',
        fontSize: "1.7rem",
        left: "1vw",
        top: "3px",
        position: "relative",
        fontWeight: 600,
        fontFamily: 'inherit',
      }}
    />
  );
}