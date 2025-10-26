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
import { OwnershipBar } from './Ownership';
import { SearchBar, searchFilter } from './SearchBar';
import { NeuralNetworkInfo } from './NeuralNetworks';
import { getNeuralNetworks } from './NeuralNetworks';
import CardActionArea from '@mui/material/CardActionArea';
import ClearIcon from '@mui/icons-material/Clear';
import DeleteButton from './DeleteButton';

export default function dashboardBody() {
    const [NeuralNetworks, setNeuralNetworks] = React.useState<NeuralNetworkInfo[]>(getNeuralNetworks()); // do not use setNeuralNetworks as that is the master
    const [visibleNetworks, setVisibleNetworks] = React.useState<NeuralNetworkInfo[]>(NeuralNetworks);  //copy of neuralnetworks that gets decimated

    const handleSortChange = (sortType: string) => {
        setVisibleNetworks(NewSort(sortType, getNeuralNetworks())); //Passes full neural network array to newSort
    };

    const handleSearch = (input: string) => {
        setVisibleNetworks(searchFilter(input, getNeuralNetworks())); //Passes full neural network array to searchFilter
    };

    return (
        <Container sx={{ bgcolor: "#EDF1F3", minHeight: "100vh", minWidth: "100vw"}}>
            <>
                {/* NavBar */}
                <Box sx={{ display: "flex", flexWrap: "wrap", p: 4, justifyContent:"space-between"}}>
                    <SearchBar stateChanger={handleSearch}/>
                    {/* passes handlestatechange to child so it can re-render parent (this) */}
                    <SortingBar stateChanger={handleSortChange}/> 
                    {/* TODO: same thing as sorting bar */}
                    <OwnershipBar/>
                </Box>

                {/* Neural Network, adds them in in the order of cards array*/}
                <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2, justifyContent: "center" }}>
                    {visibleNetworks.map((NeuralNetwork ,index) => (
                        <Card key={index} sx={{ maxWidth: 800 }}>
                            <CardActionArea>
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
                                            <Typography variant="body2" sx={{ color: 'text.secondary', flexGrow: 1}}>
                                                Last Accessed : {NeuralNetwork.lastAccessed}
                                            </Typography>
                                        </CardContent>
                                        <CardActions>
                                            <FavouriteButton/>
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
