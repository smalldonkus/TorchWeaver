"use client";
// Import MUI theme hook for accessing theme variables (like direction)
import { useTheme } from "@mui/material/styles";
// Import Material UI Drawer component for sidebar
import Drawer from "@mui/material/Drawer";
// Divider for separating sections visually
import Divider from "@mui/material/Divider";
// IconButton for clickable icons (like closing the sidebar)
import IconButton from "@mui/material/IconButton";
// List and ListItem components for menu structure
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import InputIcon from '@mui/icons-material/Input';
import OutputIcon from '@mui/icons-material/Output';
import ApiIcon from '@mui/icons-material/Api';

// Import icons for menu items
import UploadIcon from "@mui/icons-material/Upload";
import SaveIcon from "@mui/icons-material/Save";
import LayersIcon from "@mui/icons-material/Layers";
import FunctionsIcon from "@mui/icons-material/Functions";
import EditIcon from "@mui/icons-material/Edit";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";

// Import styled DrawerHeader and drawerWidth constant for consistent styling
import { DrawerHeader } from "../utils/styled";
import { drawerWidth } from "../utils/constants";
// Import LayerForm component for the "Layers" menu
import LayerForm from "./LayerForm";
// Import TensorOpsForm component for the "Tensor Operations" menu
import TensorOpsForm from "./TensorOpsForm";
// Import InputForm component for the "Inputs" menu
import InputForm from "./InputForm";
// Import OutputForm component for the "Outputs" menu
import OutputForm from "./OutputForm";
// Import ActivatorsForm component for the "Activators" menu
import ActivatorsForm from "./ActivatorsForm";

// Import EditLayerForm component for the "Edit Layer" menu
import EditLayerForm from "./EditLayerForm"

// import auth0 client for dynamic rendering if a user is signed in
import { useUser } from "@auth0/nextjs-auth0"

// Define the props expected by the Sidebar component
interface Props {
    open: boolean; // Whether the sidebar is open
    setOpen: (val: boolean) => void; // Function to open/close the sidebar
    selectedMenu: string; // Currently selected menu item
    setSelectedMenu: (val: string) => void; // Function to change selected menu
    nodes: any[]; // Array of nodes (data for LayerForm)
    setNodes: (val: any) => void; // Function to update nodes
    handleSave: () => void; // Function to handle save action
    handleExport: () => void; // Function to handle export action
    selectedNodes: any[]; // shows current selected Nodes
    updateNodeType: (targetID: any, valA: any, valB: any, valC: any) => void; // allows the update of layerType
    updateNodeOperationType: (targetID: any, valA: any, valB: any, valC: any) => void;
    updateNodeParameter: (targetID: any, valA: any, valB: any) => void;
    deleteNode: (targetID: any) => void;
    getSetters: () => any;
    getDefaults: () => any;
    defaultLayers: any; // Changed from any[] to any for new structure
    defaultTensorOps: any; // Changed from any[] to any for new structure
    defaultActivators: any; // Changed from any[] to any for new structure
    defaultInputs: any; // Input definitions data structure
}

// Sidebar component definition
export default function Sidebar({
    open,
    setOpen,
    selectedMenu,
    setSelectedMenu,
    nodes,
    setNodes,
    handleExport,
    handleSave,
    selectedNodes,
    updateNodeType,
    updateNodeOperationType,
    updateNodeParameter,
    deleteNode,
    getSetters,
    getDefaults,
    defaultLayers,
    defaultTensorOps,
    defaultActivators,
    defaultInputs
}: Props) {
    // Get theme object for direction (ltr/rtl)
    const theme = useTheme();

    const { user, isLoading } = useUser();
    const userDisabling = !user;

    return (
        // Drawer component for the sidebar
        <Drawer
            sx={{
                width: drawerWidth, // Set sidebar width
                flexShrink: 0, // Prevent shrinking
                "& .MuiDrawer-paper": {
                    width: drawerWidth, // Set paper width
                    display: "flex", // Use flex layout
                    flexDirection: "column", // Stack children vertically
                    justifyContent: "space-between", // Space between top and bottom sections
                },
            }}
            variant="persistent" // Sidebar stays open until closed
            anchor="left" // Sidebar appears on the left
            open={open} // Controlled by open prop
        >
            {/* Top section of the sidebar */}
            <div>
                {/* Header with close button */}
                <DrawerHeader>
                    <IconButton onClick={() => setOpen(false)}>
                        {/* Show left or right chevron based on theme direction */}
                        {theme.direction === "ltr" ? <ChevronLeftIcon /> : <ChevronRightIcon />}
                    </IconButton>
                </DrawerHeader>
                <Divider />
                {/* Main menu list */}
                <List>
                    {/* Define menu items with text and icon */}
                    {[
                        { text: "Layers", icon: <LayersIcon /> },
                        { text: "Tensor Operations", icon: <FunctionsIcon /> },
                        { text: "Activation Functions", icon: <ApiIcon /> },
                        { text: "Inputs", icon: <InputIcon /> },
                        { text: "Outputs", icon: <OutputIcon /> },
                        { text: "Edit Nodes", icon: <EditIcon /> },
                    ].map((item) => (
                        // Render each menu item
                        <ListItem key={item.text} disablePadding>
                            <ListItemButton
                                selected={selectedMenu === item.text} // Highlight if selected
                                onClick={() => {
                                    setSelectedMenu(item.text);
                                }} // Change selected menu
                            >
                                <ListItemIcon>{item.icon}</ListItemIcon>
                                <ListItemText primary={item.text} />
                            </ListItemButton>
                        </ListItem>
                    ))}
                </List>
                {/* Show LayerForm only if "Layers" menu is selected */}
                {selectedMenu === "Layers" && (
                    <LayerForm nodes={nodes} setNodes={setNodes} defaultLayers={defaultLayers} getSetters={getSetters} getDefaults={getDefaults}/>
                )}
                {selectedMenu === "Edit Nodes" && (
                    <EditLayerForm selectedNodes={selectedNodes} defaultActivators={defaultActivators} defaultTensorOps={defaultTensorOps} defaultLayers={defaultLayers} defaultInputs={defaultInputs} updateNodeType={updateNodeType} updateNodeOperationType={updateNodeOperationType} updateNodeParameter={updateNodeParameter} deleteNode={deleteNode}/>
                )}
                {/* Show TensorOpsForm only if "Tensor Operations" menu is selected */}
                {selectedMenu === "Tensor Operations" && (
                    <TensorOpsForm nodes={nodes} setNodes={setNodes} defaultTensorOps={defaultTensorOps} getSetters={getSetters} getDefaults={getDefaults} />
                )}
                {selectedMenu === "Inputs" && (
                    <InputForm nodes={nodes} setNodes={setNodes} defaultInputs={defaultInputs} getSetters={getSetters} getDefaults={getDefaults}/>
                )}
                {selectedMenu === "Outputs" && (
                    <OutputForm nodes={nodes} setNodes={setNodes} getSetters={getSetters} getDefaults={getDefaults}/>
                )}
                {selectedMenu === "Activation Functions" && (
                    <ActivatorsForm nodes={nodes} setNodes={setNodes} defaultActivators={defaultActivators} getSetters={getSetters} getDefaults={getDefaults}/>
                )}
            </div>
            {/* Bottom section of the sidebar */}
            <div>
                <Divider />
                {/* List for export and save actions */}
                <List>
                    {/* Export button */}
                    <ListItem disablePadding>
                        <ListItemButton onClick={handleExport} disabled={userDisabling}>
                            <ListItemIcon>
                                <UploadIcon />
                            </ListItemIcon>
                            <ListItemText primary="Export" />
                        </ListItemButton>
                    </ListItem>
                    {/* Save button (no handler attached) */}
                    <ListItem disablePadding>
                        <ListItemButton onClick={handleSave} disabled={userDisabling}>
                            <ListItemIcon>
                                <SaveIcon />
                            </ListItemIcon>
                            <ListItemText primary="Save" />
                        </ListItemButton>
                    </ListItem>
                </List>
            </div>
        </Drawer>
    );
}
