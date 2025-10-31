"use client";

import * as React from 'react';
import { Box, Container } from "@mui/material"
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import CardActions from '@mui/material/CardActions';
import CardMedia from '@mui/material/CardMedia';
import { BackToTopButton } from "./BackToTopButton";
import { FavouriteButton } from './FavouriteButton';
import { NewSort, SortingBar } from './Sorting';
import { NewList, OwnershipBar } from './Ownership';
import { SearchBar, searchFilter } from './SearchBar';
import { NeuralNetworkInfo } from './NeuralNetworks';
import { getNeuralNetworks } from './NeuralNetworks';
import CardActionArea from '@mui/material/CardActionArea';
import DeleteButton from './DeleteButton';

export default function dashboardBody() {
    const [NeuralNetworks, setNeuralNetworks] = React.useState<NeuralNetworkInfo[]>(getNeuralNetworks()); // do not use setNeuralNetworks as that is the master
    const [visibleNetworks, setVisibleNetworks] = React.useState<NeuralNetworkInfo[]>(NeuralNetworks);  //copy of neuralnetworks that gets decimated

    // helper functions to ensure favourited networks appear first
    const handleFavourites = (networks: NeuralNetworkInfo[]) => {
        const favourited = getNeuralNetworks().filter(nn => nn.Favourited);
        const nonFavourited = getNeuralNetworks().filter(nn => !nn.Favourited);
        return [...favourited, ...nonFavourited];
    };

    const handleFavourite = (index: number, newState: boolean) => {
        console.log("debug: favourites = ", newState);

        const updated = [...getNeuralNetworks()];
        updated[index].Favourited = newState;
        setVisibleNetworks(updated);
    };

    const handleSortChange = (sortType: string) => {
        setVisibleNetworks(NewSort(sortType, handleFavourites(getNeuralNetworks()))); //Passes full neural network array to newSort
    };

    const handleSearch = (input: string) => {
        setVisibleNetworks(searchFilter(input, handleFavourites(getNeuralNetworks()))); //Passes full neural network array to searchFilter
    };

    const handleOwnershipSorting = (sortType: string) => {
        const owner = "A";
        setVisibleNetworks(NewList(owner, sortType, handleFavourites(getNeuralNetworks())));
    };

    return (
        <Container sx={{ bgcolor: "#EDF1F3", minHeight: "100vh", minWidth: "100vw"}}>
            <>
                {/* NavBar */}
                <Box sx={{ display: "flex", flexWrap: "wrap", p: 4, justifyContent:"space-between"}}>
                    <SearchBar stateChanger={handleSearch}/>
                    {/* passes handlestatechange to child so it can re-render parent (this) */}
                    <SortingBar stateChanger={handleSortChange}/> 
                    <OwnershipBar stateChanger={handleOwnershipSorting}/>
                </Box>

                {/* Neural Network, adds them in in the order of cards array*/}
                <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2, justifyContent: "center" }}>
                    {visibleNetworks.map((NeuralNetwork ,index) => (
                        <Card key={index} sx={{ maxWidth: 800 }}>
                            <CardActionArea href='/canvas'>
                                <CardMedia
                                    component="img"
                                    height="194"
                                    image={NeuralNetwork.image}
                                    alt={NeuralNetwork.title}
                                />
                            </CardActionArea>
                                {/* Align heart and words */}
                                <Box sx={{display: "flex", justifyContent: "space-between"}}>
                                        <CardContent>
                                            <Typography variant="h6" sx={{ color: 'text' , flexGrow: 1}}>{NeuralNetwork.title}</Typography>
                                            <Typography variant="body2" sx={{ color: 'text.secondary', flexGrow: 1 }}>
                                                <span>Last Accessed: {NeuralNetwork.lastAccessed}</span>
                                                <span style={{ marginLeft: 30 }}>Owned By: {NeuralNetwork.Owner}</span>
                                            </Typography>
                                        </CardContent>
                                        <CardActions>
                                            <FavouriteButton 
                                                isFavourtied={NeuralNetwork.Favourited}
                                                onToggle={(newState) => handleFavourite(index, newState)}
                                            />
                                            <DeleteButton/>
                                        </CardActions>
                                </Box>
                        </Card>
                    ))}
                </Box>  
            </>

            <BackToTopButton/>
        </ Container>
    )
}
