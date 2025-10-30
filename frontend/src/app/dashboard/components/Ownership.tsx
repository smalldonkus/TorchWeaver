"use client";
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select, { SelectChangeEvent } from "@mui/material/Select";
import MenuItem from '@mui/material/MenuItem';
import * as React from 'react';
import { NeuralNetworkInfo } from './NeuralNetworks';

export const OwnershipBar = ({stateChanger}) => {
    const [Ownership, setOwner] = React.useState("OwnedByAnyone");

    const handleChange = (event: SelectChangeEvent) => {
        const owner = event.target.value;
        setOwner(owner);
        stateChanger(owner);
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

export function NewList(owner: string, sortType: string, NeuralNetworks: NeuralNetworkInfo[]): NeuralNetworkInfo[] {
    // if(sortType === "OwnedByAnyone") {
    //     return NeuralNetworks;
    // }
    if(sortType === "OwnedByMe") {
        return NeuralNetworks.filter(n => n.Owner === owner);
    }
    if(sortType === "NotOwnedByMe") {
        return NeuralNetworks.filter(n => n.Owner !== owner);
    }
    return NeuralNetworks;
}
