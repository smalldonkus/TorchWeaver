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
import { NeuralNetworkInfo, getNeuralNetworks, loadNetwork, deleteNetwork } from './NeuralNetworks';
import { useUser } from '@auth0/nextjs-auth0/client';
import CardActionArea from '@mui/material/CardActionArea';
import DeleteButton from './DeleteButton';

export default function dashboardBody() {
    const [networks, setNetworks] = React.useState<NeuralNetworkInfo[]>([]);
    const [visibleNetworks, setVisibleNetworks] = React.useState<NeuralNetworkInfo[]>([]);
    const [loading, setLoading] = React.useState(true);
    const [error, setError] = React.useState<string | null>(null);

    const { user } = useUser();

    React.useEffect(() => {
        const load = async () => {
            try {
                setLoading(true);
                if (!user) {
                    setError('Please sign in to view your saved networks.');
                    setNetworks([]);
                    setVisibleNetworks([]);
                    return;
                }

                const userId = user.sub;
                const data = await getNeuralNetworks(userId);
                setNetworks(data);
                setVisibleNetworks(data);
                setError(null);
            } catch (err) {
                setError('Failed to load networks');
                console.error(err);
            } finally {
                setLoading(false);
            }
        };
        load();
    }, [user]);

    const handleSortChange = (sortType: string) => {
        setVisibleNetworks(NewSort(sortType, networks));
    };

    const handleSearch = (input: string) => {
        setVisibleNetworks(searchFilter(input, networks));
    };

    const handleNetworkClick = async (id: number) => {
        try {
            const userId = user?.sub;
            const net = await loadNetwork(id, userId);
            // Keep the loaded network in localStorage for backward compatibility with older flow
            localStorage.setItem('loadedNetwork', JSON.stringify(net));
            // Redirect to canvas with the saved network id so the canvas page can fetch it
            window.location.href = `/canvas?id=${id}`;
        } catch (err) {
            console.error('Failed to load network', err);
        }
    };

    const handleDelete = async (id: number) => {
        if (!confirm('Delete this network?')) return;
        try {
            const userId = user?.sub;
            await deleteNetwork(id, userId);
            const updated = await getNeuralNetworks(userId);
            setNetworks(updated);
            setVisibleNetworks(updated);
        } catch (err) {
            console.error('Failed to delete network', err);
        }
    };
    //ownership sorting and searching are both destructive only one can happen at a time
    //Oscar note : i dont think this is fixable, sad
    const handleOwnershipSorting = async (sortType: string) => {
        const owner = "A";
        const userId = user?.sub;
        const nets = await getNeuralNetworks(userId);
        setVisibleNetworks(NewList(owner, sortType, nets));
    };

    //function to change favourited boolean in global if toggled
    const handleFavourite = (title: string) => {
        const updated = visibleNetworks.map((net) =>
            net.title === title ? { ...net, Favourited: !net.Favourited } : net
        );
        setVisibleNetworks(updated); // Updates visuals
        // setNeuralNetworks(updated); // stores back into global
    }

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

                {loading && (
                    <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}><Typography>Loading networks...</Typography></Box>
                )}

                {error && (
                    <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}><Typography color="error">{error}</Typography></Box>
                )}

                {/* Neural Network grid */}
                <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2, justifyContent: "center" }}>
                    {visibleNetworks.map((network) => (
                        <Card key={network.id} sx={{ maxWidth: 800 }}>
                            <CardActionArea onClick={() => handleNetworkClick(network.id)}>
                                <CardMedia
                                    component="img"
                                    height="194"
                                    image={network.image}
                                    alt={network.title}
                                />
                            </CardActionArea>
                                {/* Align heart and words */}
                                <Box sx={{display: "flex", justifyContent: "space-between"}}>
                                        <CardContent>
                                            <Typography variant="h6" sx={{ color: 'text' , flexGrow: 1}}>{network.title}</Typography>
                                            <Typography variant="body2" sx={{ color: 'text.secondary', flexGrow: 1 }}>
                                                <span>Last Accessed: {network.lastAccessed}</span>
                                                <span style={{ marginLeft: 30 }}>Owned By: {network.Owner}</span>
                                            </Typography>
                                        </CardContent>
                                        <CardActions>
                                            <FavouriteButton
                                                isFavourite={network.Favourited}
                                                onClick={() => handleFavourite(network.title)}/>
                                            <DeleteButton onClick={() => handleDelete(network.id)}/>
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
