"use client";

import React, { useState } from "react";
import { IconButton } from "@mui/material";
import FavoriteIcon from "@mui/icons-material/Favorite";

interface FavouriteButtonProps {
    isFavourtied: boolean;
    onToggle: (newState: boolean) => void;
}

export const FavouriteButton = ({ isFavourtied, onToggle }: FavouriteButtonProps) => {
    const [current, setLiked] = useState(isFavourtied); //Stores state and function that updates state

    const click = () => {
        // setLiked(!current); // on click, set state to not current state 
        const newState = !current;
        setLiked(newState);
        onToggle(newState);
    };

    return (
        <IconButton aria-label="add to favorites"
        onClick={click}
        sx={{
        color: current ? "red" : "grey",
        transition: "color 0.3s ease",
    }}
    >
    <FavoriteIcon />
    </IconButton>
  );
};
