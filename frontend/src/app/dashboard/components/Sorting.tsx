"use client";
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select, { SelectChangeEvent } from "@mui/material/Select";
import MenuItem from '@mui/material/MenuItem';
import * as React from 'react';
import { NeuralNetworkInfo } from './NeuralNetworks';

export const SortingBar = ({sorting, stateChanger}) => {

    const handleChange = (event: SelectChangeEvent) => {
        const sortType = event.target.value 
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
    if (sortType === "Oldest") {
    return NeuralNetworks.toSorted((a, b) => {
        const timeA = parseCustomDate(a.lastAccessed).getTime();
        const timeB = parseCustomDate(b.lastAccessed).getTime();
        return timeA - timeB;
    });
    ;
}
    if (sortType === "Newest") {
    return NeuralNetworks.toSorted((a, b) => {
        const timeA = parseCustomDate(a.lastAccessed).getTime();
        const timeB = parseCustomDate(b.lastAccessed).getTime();
        return timeB - timeA;
    });
}
    return NeuralNetworks;
}


function parseCustomDate(str: string): Date {
    // Split into parts like ["06/11/2025", "2:50:41", "pm"]
    const [datePart, timePart, ampm] = str.split(/[, ]+/);
    const [day, month, year] = datePart.split('/').map(Number);
    let [hour, minute, second] = timePart.split(':').map(Number);

    const isPM = ampm.toLowerCase() === 'pm';
    if (ampm.toLowerCase() === 'pm' && hour < 12) hour += 12;
    if (ampm.toLowerCase() === 'am' && hour === 12) hour = 0;

    return new Date(year, month - 1, day, hour, minute, second);
}