"use client";

import React, { useState } from "react";
import { IconButton } from "@mui/material";
import FavoriteIcon from "@mui/icons-material/Favorite";

export const FavouriteButton = () => {
    const [current, setLiked] = useState(false); //Stores state and function that updates state

    const click = () => {
    setLiked(!current); // on click, set state to not current state 
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
