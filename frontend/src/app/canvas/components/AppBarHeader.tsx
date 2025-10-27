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
// Import a styled AppBar component
import { AppBar as StyledAppBar } from "../utils/styled";

// Define the props for this component
interface Props {
    open: boolean; // Whether the sidebar/drawer is open
    setOpen: (val: boolean) => void; // Function to set the open state
}

// Main AppBarHeader component
export default function AppBarHeader({ open, setOpen }: Props) {
    // State to control the anchor element for the dropdown menu
    const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
    // Next.js router for navigation
    const router = useRouter();

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
                <Typography variant="h6" noWrap sx={{ flexGrow: 1 }}>
                    TorchWeaver Canvas
                </Typography>
                {/* "Return" button that opens the dropdown menu */}
                <Button color="inherit" onClick={handleMenuClick}>
                    Return
                </Button>
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
                            router.push("/dashboard");
                        }}
                    >
                        Return to Dashboard
                    </MenuItem>
                    {/* Menu item to log out */}
                    <MenuItem
                        onClick={() => {
                            handleMenuClose();
                            router.push("/login");
                        }}
                    >
                        Logout
                    </MenuItem>
                </Menu>
            </Toolbar>
        </StyledAppBar>
    );
}
