"use client";

import * as React from 'react';
import TextField from '@mui/material/TextField';
import InputAdornment from '@mui/material/InputAdornment';
import { Search } from '@mui/icons-material';
import { NeuralNetworkInfo } from './NeuralNetworks';

export const SearchBar = ({stateChanger}) => { 
    const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        stateChanger(event.target.value); //sends input back to parent
    };
    
    return (
        <TextField id="searchBar" label="Search" variant="filled" sx={{minWidth: "50vw"}} size="small"
                slotProps={{
                input: {
                startAdornment: (
                <InputAdornment position="start">
                    <Search/>
                </InputAdornment> //Search bar with icon
                )}
        }} onChange={handleChange}/>
    )
}

export function searchFilter(input: string, NeuralNetworks: NeuralNetworkInfo[]): NeuralNetworkInfo[] {
    const copy = NeuralNetworks; // must create copy to not destroy master
    return copy.filter((nn :NeuralNetworkInfo) => nn.title.toLowerCase().includes(input.toLowerCase()))
}