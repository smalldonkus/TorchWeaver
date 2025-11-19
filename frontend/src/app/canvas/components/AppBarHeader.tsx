"use client";
// Import React and hooks
import { useState } from "react";
// Import Next.js router for navigation
import { useRouter } from "next/navigation";
// Import Material UI components
import MuiAppBar, { AppBarProps as MuiAppBarProps } from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import IconButton from "@mui/material/IconButton";
import Button from "@mui/material/Button";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Fade from "@mui/material/Fade";
import MenuIcon from "@mui/icons-material/Menu";
import UndoIcon from '@mui/icons-material/Undo';
import RedoIcon from '@mui/icons-material/Redo';
// Import a styled AppBar component
import { AppBar as StyledAppBar } from "../utils/styled";
import { Box } from "@mui/material";
import NamingBox from "./NamingBox";

// Define the props for this component
interface Props {
    open: boolean; // Whether the sidebar/drawer is open
    setOpen: (val: boolean) => void; // Function to set the open state
    doUndo: () => void;
    doRedo: () => void;
    name: string
    setName: React.Dispatch<React.SetStateAction<string>>
    hasUnsavedChanges?: boolean; // Whether there are unsaved changes
    onNavigate?: (url: string) => void; // Custom navigation handler
}

// Main AppBarHeader component
export default function AppBarHeader({ open, setOpen, doUndo, doRedo, name, setName, hasUnsavedChanges, onNavigate}: Props) {
    // State to control the anchor element for the dropdown menu
    const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
    // Next.js router for navigation
    const router = useRouter();
    
    // Handle navigation with unsaved changes check
    const handleNavigation = (url: string) => {
        if (onNavigate && hasUnsavedChanges) {
            onNavigate(url);
        } else {
            router.push(url);
        }
    };

    // When the "Return" button is clicked, open the menu
    const handleMenuClick = (e: React.MouseEvent<HTMLElement>) =>
        setAnchorEl(e.currentTarget);
    // Close the menu
    const handleMenuClose = () => setAnchorEl(null);

    return (
        // The top app bar, styled and fixed position
        <StyledAppBar position="fixed" open={open}>
            <Toolbar>
                {/* Hamburger menu icon, only shown when sidebar is closed */}
                <IconButton
                    color="inherit"
                    aria-label="open drawer"
                    onClick={() => setOpen(true)}
                    edge="start"
                    sx={{ mr: 2, ...(open && { display: "none" }) }}
                >
                    <MenuIcon />
                </IconButton>
                {/* App title */}
                {/* Clickable Logo */}
                <Box
                    component="img"
                    src="/7945d26d-5a29-472d-905f-c96a4022f7ef.png"
                    alt="Torchweaver logo"
                    sx={{ height: 40, cursor: 'pointer' }}
                    onClick={() => handleNavigation('/')} // navigate back to home when logo is pressed
                />
                <NamingBox value={name} onChange={setName}/>
                {/* Undo/Redo and Return buttons grouped on the right */}
                <Box
                    sx={{
                        position: "fixed",
                        right: "20px",
                        display: "flex",
                        alignItems: "center",
                        gap: "10px"
                    }}
                >
                    {/* "Undo" button that undoes the last "significant" action */}
                    <Box
                        sx={{
                            display: "flex", 
                            flexDirection:"row", 
                            gap: "10px",
                            border:"1px dashed white",
                            borderRadius: "10px",
                            alignItems: "center"
                        }}>
                        <IconButton
                            color="inherit"
                            aria-label="undo"
                            onClick={doUndo}
                            edge="end"
                            sx={{ml: 0.5, mr: 0.5}}
                        >
                            <UndoIcon />
                        </IconButton>
                        <IconButton
                            color="inherit"
                            aria-label="redo"
                            onClick={doRedo}
                            edge="end"
                            sx={{mr: 0.5}}
                        >
                            <RedoIcon />
                        </IconButton>
                    </Box>
                    {/* "Return" button that opens the dropdown menu */}
                    <Button 
                        onClick={handleMenuClick}
                        sx={{
                            backgroundColor: "#FF7700",
                            color: "white",
                            fontSize: "1.1rem",
                            borderRadius: "8px",
                            fontWeight: 600,
                            fontFamily: 'inherit',
                            textTransform: 'none',
                            '&:hover': {
                                backgroundColor: '#f88f34ff',
                            }
                        }}
                    >
                        Return
                    </Button>
                </Box>
                {/* Dropdown menu with navigation options */}
                <Menu
                    anchorEl={anchorEl}
                    open={Boolean(anchorEl)}
                    onClose={handleMenuClose}
                    TransitionComponent={Fade}
                >
                    {/* Menu item to go back to dashboard */}
                    <MenuItem
                        onClick={() => {
                            handleMenuClose();
                            handleNavigation("/dashboard");
                        }}
                    >
                        Return to Dashboard
                    </MenuItem>
                    {/* Menu item to log out */}
                    <MenuItem
                        onClick={() => {
                            handleMenuClose();
                            handleNavigation("/login");
                        }}  
                    >
                        Logout
                    </MenuItem>
                </Menu>
            </Toolbar>
        </StyledAppBar>
    );
}

// {/* "Undo" button that undoes the last "significant" action */}
//                 <Box
//                     sx={{
//                         display: "flex", 
//                         flexDirection:"row", 
//                         gap: "10px",
//                         border:"1px dashed white",
//                         borderRadius: "10px",
//                         alignItems: "center",
//                         mr:2
//                     }}>
//                     <IconButton
//                         color="inherit"
//                         aria-label="open drawer"
//                         onClick={doUndo}
//                         edge="end"
//                         sx={{ml: 0.5, mr: 0.5}}
//                     >
//                         <UndoIcon />
//                     </IconButton>
//                     <IconButton
//                         color="inherit"
//                         aria-label="open drawer"
//                         onClick={doRedo}
//                         edge="end"
//                         sx={{mr: 0.5}}
//                     >
//                         <RedoIcon />
//                     </IconButton>
//                 </Box>
//                 {/* "Error" button that opens the Errors drawer  */}
//                 <Button variant={errorButtonVariant} color="error" onClick={handleErrorClick} sx={{mr:1}}>
//                     Errors
//                 </Button>