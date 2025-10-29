import { Box, Drawer, List, ListItem, ListItemIcon, ListItemText } from "@mui/material";
import ErrorIcon from '@mui/icons-material/Error';

import { useEffect, useState } from "react";

interface Props {
    isOpen: boolean;
    setOpen: (val: any) => void;
    messages: string[];
}

export default function ErrorBox({isOpen, setOpen, messages}: Props){
    
    /*
        when an error occurs, a box should appear on top of all
        other elements in the BOTTOM RIGHT.

        inside should be collapsable ACCORDION elements, with a
        number for each error.
    */


    const toggleDrawer =
        (open: boolean) =>
        (event: React.KeyboardEvent | React.MouseEvent) => {
            // event allows you to stop the close on certain inputs
            // ignore for now.
            setOpen(open);
    };

    const list = () => (
        <Box
            sx={{width: 'auto', alignContent:"center"}}
            role="presentation"
            key={"errorBox"}
        >
            {messages.length == 0 && (
                <List key={"no messages"}>
                    <ListItem key={"no messages"}>
                        <ListItemIcon>
                            <ErrorIcon/>
                        </ListItemIcon>
                        <ListItemText primary={"No Errors!"}/>
                    </ListItem>
                </List>
            )}
            {messages.length != 0 && 
                messages.map((msg, index) => (
                    <List key={index}>
                        <ListItem key={index}>
                            <ListItemIcon>
                                <ErrorIcon color="error"/>
                            </ListItemIcon>
                            <ListItemText primary={msg}/>
                        </ListItem>
                    </List>
                ))
            }
        </Box>
    );

    const anchor = 'bottom';
    return (
        <Drawer
            anchor={anchor}
            open={isOpen}
            onClose={toggleDrawer(false)}
        >
            {list()}
        </Drawer>
    )
}