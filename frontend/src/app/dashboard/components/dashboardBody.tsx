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
import Fab from '@mui/material/Fab';
import AddIcon from '@mui/icons-material/Add';

export default function dashboardBody() {
    const [networks, setNetworks] = React.useState<NeuralNetworkInfo[]>([]); //Don't access networks, unsure of what it does (probably nothing as i assume its local to this file)
    const [visibleNetworks, setVisibleNetworks] = React.useState<NeuralNetworkInfo[]>([]);
    const [sorting, setSort] = React.useState("Alphabetical");
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
                setVisibleNetworks(NewSort("Alphabetical", data));  //Sets initial state of sort to A-Z
                
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

    const handleNew = async () => {
        try {
            window.location.href = `/canvas`;
        } catch (err) {
            console.error('Error creating new network', err);
        }
    }

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

    const handleSearch = async (input: string) => {
        try {
            const userId = user?.sub;
            const networks = await getNeuralNetworks(userId);
            const filtered = searchFilter(input, networks);
            setVisibleNetworks(NewSort(sorting, filtered));
        } catch (err) {
            console.error('Failed to retrieve networks', err);
        }
    };

    const handleSortChange = async (sortType: string) => {
        try {
            const userId = user?.sub;
            const networks = await getNeuralNetworks(userId);
            setSort(sortType);
            setVisibleNetworks(NewSort(sortType, networks));
        } catch (err) {
            console.error('Failed to retrieve networks', err);
        }
    };
    //ownership sorting and searching are both destructive only one can happen at a time
    //Oscar note : i dont think this is fixable, sad
    const handleOwnershipSorting = async (sortType: string) => {
        const owner = "User";
        const userId = user?.sub;
        const nets = await getNeuralNetworks(userId);
        setVisibleNetworks(NewList(owner, sortType, nets));
    };

    //favourite neural network, favourited networks show up first regardless of sorting (sorts 2 lists and appends unfavourited to favourited)
    const handleFavourite = async (networkId: number) => {

        const updated = visibleNetworks.map((net) =>
            net.id === networkId ? { ...net, Favourited: !net.Favourited } : net //Finds network and toggles favourite   
        );
        setVisibleNetworks(updated); // Updates visuals
        try {
            await fetch('http://localhost:5000/favourite_network', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json'},
                body: JSON.stringify({ id: networkId, favourited: updated.find(n => n.id === networkId)?.Favourited}) //changes the status of the favourite in the db
            });
        }
        catch (err) {
        console.error('Failed to update favourite status', err);
        setVisibleNetworks(visibleNetworks); //Revert if failed
    }
    }

    return (
        <Container sx={{ bgcolor: "#EDF1F3", minHeight: "100vh", minWidth: "100vw"}}>
            <>
                {/* NavBar */}
                <Box sx={{ display: "flex", flexWrap: "wrap", p: 4, justifyContent:"space-between"}}>
                    <SearchBar stateChanger={handleSearch}/>
                    {/* passes handlestatechange to child so it can re-render parent (this) */}
                    <SortingBar sorting={sorting} stateChanger={handleSortChange}/> 
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
                        <Card key={network.id} sx={{ maxWidth: 750 }}>
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
                                                onClick={() => handleFavourite(network.id)}/>
                                            <DeleteButton onClick={() => handleDelete(network.id)}/>
                                        </CardActions>
                                </Box>
                        </Card>
                    ))}
                </Box>
            </>
            <Fab color="primary" aria-label="add" onClick={() => handleNew()} sx={{
                position: "fixed",     // stay fixed on the screens
                bottom: 35,            // 35px from bottom
                left: 35,               // 35px from right
                zIndex: 1000           // make sure it appears above other content
            }}>
                <AddIcon />
            </Fab>
            <BackToTopButton/>
        </ Container>
    )
}
