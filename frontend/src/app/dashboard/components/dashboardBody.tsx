import * as React from 'react';
import { Box, Container } from "@mui/material"
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import CardActions from '@mui/material/CardActions';
import CardMedia from '@mui/material/CardMedia';
import IconButton from '@mui/material/IconButton';
import FavoriteIcon from '@mui/icons-material/Favorite';
import TextField from '@mui/material/TextField';
import FormControl from '@mui/material/FormControl';
import NativeSelect from '@mui/material/NativeSelect';
import InputLabel from '@mui/material/InputLabel';
import InputAdornment from '@mui/material/InputAdornment';
import { BackToTopButton } from "./BackToTopButton";
import { FavouriteButton } from './FavouriteButton';

import Link from "next/link"
import { Search } from '@mui/icons-material';

const cards = [
  { title: "Test Neural Network", lastAccessed: "100/100/100", image: "/testnetwork.png" },
  { title: "Test Neural Network", lastAccessed: "100/100/100", image: "/testnetwork.png" },
  { title: "Test Neural Network", lastAccessed: "100/100/100", image: "/testnetwork.png" },
];

export default function dashboardBody() {

    return (
        <Container sx={{ bgcolor: "#EDF1F3", minHeight: "100vh", minWidth: "100vw"}}>
            <>
                {/* NavBar */}
                <Box sx={{ display: "flex", flexWrap: "wrap", p: 4, justifyContent:"space-between"}}>
                    <TextField id="SearchBar" label="Search" variant="filled" sx={{minWidth: "50vw"}} slotProps={{
                        input: {
                        startAdornment: (
                        <InputAdornment position="start">
                            <Search />
                        </InputAdornment> //Search bar with icon
                        ),
                    },
                }}/>
                    <FormControl variant="filled" sx={{minWidth: "20vw"}}>
                        <InputLabel variant="standard">Sorting</InputLabel>
                        <NativeSelect defaultValue="Title">
                            <option>Title</option>
                            <option>Oldest Accessed</option>
                            <option>Newest Accessed</option>
                            <option>Oldest Accessed by me</option>
                            <option>Newest Accessed by me</option>
                        </NativeSelect>
                    </FormControl>
                    {/* Ownership dropdown */}

                    <FormControl variant="filled" sx={{minWidth: "20vw"}}>
                        <InputLabel variant="standard">Ownership</InputLabel>
                        <NativeSelect defaultValue="Owned By anyone">
                            <option>Owned by anyone</option>
                            <option>Owned by me </option>
                            <option>Not owned by me</option>
                        </NativeSelect>
                    </FormControl>
                    {/* Sorting dropdown */}
                </Box>

                {/* Neural Network, adds them in in the order of cards array*/}
                <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2, justifyContent: "center" }}>
                    {cards.map((card ,index) => (
                        <Card key={index} sx={{ maxWidth: 800 }}>
                            <CardMedia
                                component="img"
                                height="194"
                                image={card.image}
                                alt={card.title}
                            />

                            {/* Align heart and words */}
                            <Box sx={{display: "flex", justifyContent: "space-between"}}> 
                                <CardContent>
                                    <Typography variant="h6" sx={{ color: 'text' , flexGrow: 1}}>{card.title}</Typography>
                                    <Typography variant="body2" sx={{ color: 'text.secondary', flexGrow: 1}}>
                                        Last Accessed : {card.lastAccessed}
                                    </Typography>
                                </CardContent>
                                <CardActions>
                                    <FavouriteButton/>
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
