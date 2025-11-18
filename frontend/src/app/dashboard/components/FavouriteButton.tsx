"use client";

import React from "react";
import { IconButton } from "@mui/material";
import FavoriteIcon from "@mui/icons-material/Favorite";

export const FavouriteButton = ({isFavourite, onClick}) => { //Take in props
    return (
        <IconButton aria-label="add to favorites"
        onClick={onClick}
        sx={{
        color: isFavourite ? "red" : "grey",
        transition: "color 0.3s ease",
    }}
    >
    <FavoriteIcon />
    </IconButton>
  );
};
