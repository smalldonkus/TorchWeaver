"use client";

import React from "react";
import { Button } from "@mui/material";
import KeyboardArrowUpIcon from '@mui/icons-material/KeyboardArrowUp';

export const BackToTopButton = () => {
    const scrollToTop = () => {
        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });
    };

    return (
        <Button
            variant="contained"
            onClick={scrollToTop}
            startIcon={<KeyboardArrowUpIcon />}
            sx={{
                position: "fixed",
                bottom: 20,
                right: "47vw",
                zIndex: 999,
                backgroundColor: "#FF7700",
                color: "white",
                fontFamily: 'inherit',
                fontWeight: 575,
                borderRadius: '9px',
                padding: '11px 18px',
                textTransform: 'none',
                boxShadow: 3,
                "&:hover": {
                    backgroundColor: "#26489cff",
                },
            }}>
            Back to Top
        </Button>
    );
}