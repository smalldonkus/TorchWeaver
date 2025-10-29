"use client";

import React from "react";
import { Button } from "@mui/material";

export const BackToTopButton = () => {
    const scrollToTop = () => {
        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });
    };

    return (
        <Button
            variant="text"
            onClick={scrollToTop}
            sx={{
                position: "fixed",     // stay fixed on the screens
                bottom: 20,            // 20px from bottom
                right: "47vw",             // 20px from right
                zIndex: 1000           // make sure it appears above other content
            }}>
            Back to Top
        </Button>
    );
}