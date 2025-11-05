"use client";
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select, { SelectChangeEvent } from "@mui/material/Select";
import MenuItem from '@mui/material/MenuItem';
import * as React from 'react';
import { NeuralNetworkInfo } from './NeuralNetworks';

export const SortingBar = ({stateChanger}) => {
    const [sorting, setSort] = React.useState("Alphabetical");

    const handleChange = (event: SelectChangeEvent) => {
        const sortType = event.target.value 
        setSort(sortType); //Sets the sorting to the new sort to appear on screen
        stateChanger(sortType); //sends input back to parent
    };
    
    return (
        <FormControl variant="standard" sx={{minWidth: "20vw"}}>
            <InputLabel variant="standard">Sorting</InputLabel>
            <Select
            onChange={handleChange}
            value={sorting}>
                <MenuItem value="Alphabetical">A-Z</MenuItem>
                <MenuItem value="AlphabeticalR">Z-A</MenuItem>
                <MenuItem value="Oldest">First Modified</MenuItem>
                <MenuItem value="Newest">Last Modified</MenuItem>
                <MenuItem value="OldestByMe">First Modified by me</MenuItem>
                <MenuItem value="NewestByMe">Last Modified by me</MenuItem>
            </Select>
        </FormControl>
    )
}

export function NewSort(sortType: string, NeuralNetworks: NeuralNetworkInfo[]): NeuralNetworkInfo[] {
    //Split array into favourited and not favourited to be sorted twice then put back together
    const [favourited, notFavourited]: [NeuralNetworkInfo[], NeuralNetworkInfo[]] = NeuralNetworks.reduce(
        ([fav, notfav], network) => {
            if (network.Favourited) fav.push(network);
            else notfav.push(network);
            return [fav, notfav];
        },
        [[],[]] as [NeuralNetworkInfo[], NeuralNetworkInfo[]]
    )
    return LogicalSort(sortType, favourited).concat(LogicalSort(sortType, notFavourited));
}

export function LogicalSort(sortType: string, NeuralNetworks: NeuralNetworkInfo[]): NeuralNetworkInfo[] { //Updates neural networks array to correct sorting
    if(sortType === "Alphabetical") {
        return NeuralNetworks.toSorted((a, b) => a.title.localeCompare(b.title));
    }
    if(sortType === "AlphabeticalR") {
        return NeuralNetworks.toSorted((a, b) => b.title.localeCompare(a.title));
    }
    if(sortType === "Oldest") {
        return NeuralNetworks.toSorted((a, b) => {
        const [dayA, monthA, yearA] = a.lastAccessed.split('/').map(Number);
        const [dayB, monthB, yearB] = b.lastAccessed.split('/').map(Number);

        const dateNumA = yearA * 10000 + monthA * 100 + dayA;
        const dateNumB = yearB * 10000 + monthB * 100 + dayB;

        return dateNumA - dateNumB;
        });
    }
    if(sortType === "Newest") {
        return NeuralNetworks.toSorted((a, b) => {
        const [dayA, monthA, yearA] = a.lastAccessed.split('/').map(Number);
        const [dayB, monthB, yearB] = b.lastAccessed.split('/').map(Number);

        const dateNumA = yearA * 10000 + monthA * 100 + dayA;
        const dateNumB = yearB * 10000 + monthB * 100 + dayB;

        return dateNumB - dateNumA;
    });
    }
    return NeuralNetworks;
}