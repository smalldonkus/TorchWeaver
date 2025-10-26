"use client";
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select, { SelectChangeEvent } from "@mui/material/Select";
import MenuItem from '@mui/material/MenuItem';
import * as React from 'react';

export const OwnershipBar = () => {
    const [Ownership, setOwner] = React.useState("OwnedByAnyone");

    const handleChange = (event: SelectChangeEvent) => {
        const owner = event.target.value;
        setOwner(owner);
    };
    
    return (
        <FormControl variant="standard" sx={{minWidth: "20vw"}}>
            <InputLabel variant="standard">Ownership</InputLabel>
            <Select
            onChange={handleChange}
            value={Ownership}>
                <MenuItem value="OwnedByAnyone">Owned by anyone</MenuItem>
                <MenuItem value="OwnedByMe">Owned by me </MenuItem>
                <MenuItem value="NotOwnedByMe">Not owned by me</MenuItem>
            </Select>
        </FormControl>
    )
}