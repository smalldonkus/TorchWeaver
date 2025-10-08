"use client";
import * as React from "react";
import { useState, useCallback } from "react";
import { styled, useTheme } from "@mui/material/styles";
import { useRouter } from 'next/navigation';

// MUI Components
import Box from "@mui/material/Box";
import Drawer from "@mui/material/Drawer";
import MuiAppBar, { AppBarProps as MuiAppBarProps } from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import CssBaseline from "@mui/material/CssBaseline";
import List from "@mui/material/List";
import Typography from "@mui/material/Typography";
import Divider from "@mui/material/Divider";
import IconButton from "@mui/material/IconButton";
import Button from '@mui/material/Button';
import Menu from '@mui/material/Menu';
import MenuItem from '@mui/material/MenuItem';
import Fade from '@mui/material/Fade';
import TextField from '@mui/material/TextField';
import ListItem from "@mui/material/ListItem";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";

// MUI Icons
import MenuIcon from "@mui/icons-material/Menu";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import LayersIcon from "@mui/icons-material/Layers";
import FunctionsIcon from "@mui/icons-material/Functions";
import SaveIcon from "@mui/icons-material/Save";
import UploadIcon from "@mui/icons-material/Upload";
import EditIcon from "@mui/icons-material/Edit";

// React Flow
import { ReactFlow, applyNodeChanges, applyEdgeChanges, addEdge } from "@xyflow/react";
import "@xyflow/react/dist/style.css";

// Drawer width constant
const drawerWidth = 240;

// Styled components for layout
const Main = styled("main", { shouldForwardProp: (prop) => prop !== "open" })<{
  open?: boolean;
}>(({ theme, open }) => ({
  flexGrow: 1,
  padding: theme.spacing(3),
  transition: theme.transitions.create("margin", {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen,
  }),
  marginLeft: `-${drawerWidth}px`,
  ...(open && {
    transition: theme.transitions.create("margin", {
      easing: theme.transitions.easing.easeOut,
      duration: theme.transitions.duration.enteringScreen,
    }),
    marginLeft: 0,
  }),
}));

interface AppBarProps extends MuiAppBarProps {
  open?: boolean;
}

const AppBar = styled(MuiAppBar, {
  shouldForwardProp: (prop) => prop !== "open",
})<AppBarProps>(({ theme, open }) => ({
  transition: theme.transitions.create(["margin", "width"], {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen,
  }),
  ...(open && {
    width: `calc(100% - ${drawerWidth}px)`,
    marginLeft: `${drawerWidth}px`,
    transition: theme.transitions.create(["margin", "width"], {
      easing: theme.transitions.easing.easeOut,
      duration: theme.transitions.duration.enteringScreen,
    }),
  }),
}));

const DrawerHeader = styled("div")(({ theme }) => ({
  display: "flex",
  alignItems: "center",
  padding: theme.spacing(0, 1),
  ...theme.mixins.toolbar,
  justifyContent: "flex-end",
}));

// Initial nodes and edges for ReactFlow
const initialNodes = [
  { id: "n1", position: { x: 0, y: 0 }, data: { label: "Node 1" } },
  { id: "n2", position: { x: 0, y: 100 }, data: { label: "Node 2" } },
  { id: "n3", position: { x: 0, y: 200 }, data: { label: "Node 3" } },
];

const initialEdges = [{ id: "n1-n2", source: "n1", target: "n2" }];

export default function CanvasPage() {
  // Theme and router hooks
  const theme = useTheme();
  const router = useRouter();

  // Drawer open state
  const [open, setOpen] = useState(true);

  // ReactFlow nodes and edges state
  const [nodes, setNodes] = useState(initialNodes);
  const [edges, setEdges] = useState(initialEdges);

  // Layer form state
  const [inFeatures, setInFeatures] = useState("");
  const [outFeatures, setOutFeatures] = useState("");
  const [newLabel, setNewLabel] = useState("");
  const [layerType, setLayerType] = useState("Linear");

  // Sidebar menu selection state
  const [selectedMenu, setSelectedMenu] = useState("Layers");

  // Return menu state
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const menuOpen = Boolean(menuAnchor);

  // Handlers for Return menu
  const handleMenuClick = (event: React.MouseEvent<HTMLElement>) => {
    setMenuAnchor(event.currentTarget);
  };
  const handleMenuClose = () => {
    setMenuAnchor(null);
  };
  const handleReturnDashboard = () => {
    setMenuAnchor(null);
    router.push('/'); // Navigate to dashboard
  };
  const handleLogout = () => {
    setMenuAnchor(null);
    // Add your logout logic here
    router.push('/login'); // Navigate to login
  };

  // ReactFlow change handlers
  const onNodesChange = useCallback(
    (changes) => setNodes((nodesSnapshot) => applyNodeChanges(changes, nodesSnapshot)),
    []
  );
  const onEdgesChange = useCallback(
    (changes) => setEdges((edgesSnapshot) => applyEdgeChanges(changes, edgesSnapshot)),
    []
  );
  const onConnect = useCallback(
    (params) => setEdges((edgesSnapshot) => addEdge(params, edgesSnapshot)),
    []
  );

  return (
    <Box sx={{ display: "flex" }}>
      <CssBaseline />
      {/* AppBar with Return menu */}
      <AppBar position="fixed" open={open}>
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            onClick={() => setOpen(true)}
            edge="start"
            sx={{
              mr: 2,
              ...(open && { display: "none" }),
            }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            TorchWeaver Canvas
          </Typography>
          {/* Return Button with Dropdown */}
          <Button
            id="return-button"
            aria-controls={menuOpen ? 'return-menu' : undefined}
            aria-haspopup="true"
            aria-expanded={menuOpen ? 'true' : undefined}
            onClick={handleMenuClick}
            color="inherit"
            sx={{ ml: 2 }}
          >
            Return
          </Button>
          <Menu
            id="return-menu"
            anchorEl={menuAnchor}
            open={menuOpen}
            onClose={handleMenuClose}
            TransitionComponent={Fade}
            MenuListProps={{
              'aria-labelledby': 'return-button',
            }}
          >
            <MenuItem onClick={handleReturnDashboard}>Return to Dashboard</MenuItem>
            <MenuItem onClick={handleLogout}>Logout</MenuItem>
          </Menu>
        </Toolbar>
      </AppBar>
      {/* Sidebar Drawer */}
      <Drawer
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          "& .MuiDrawer-paper": {
            width: drawerWidth,
            boxSizing: "border-box",
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between",
          },
        }}
        variant="persistent"
        anchor="left"
        open={open}
      >
        <div>
          <DrawerHeader>
            <IconButton onClick={() => setOpen(false)}>
              {theme.direction === "ltr" ? <ChevronLeftIcon /> : <ChevronRightIcon />}
            </IconButton>
          </DrawerHeader>
          <Divider />
          {/* Sidebar Menu */}
          <List>
            <ListItem disablePadding>
              <ListItemButton
                selected={selectedMenu === "Layers"}
                onClick={() => setSelectedMenu("Layers")}
              >
                <ListItemIcon>
                  <LayersIcon />
                </ListItemIcon>
                <ListItemText primary="Layers" />
              </ListItemButton>
            </ListItem>
            <ListItem disablePadding>
              <ListItemButton
                selected={selectedMenu === "Tensor Operations"}
                onClick={() => setSelectedMenu("Tensor Operations")}
              >
                <ListItemIcon>
                  <FunctionsIcon />
                </ListItemIcon>
                <ListItemText primary="Tensor Operations" />
              </ListItemButton>
            </ListItem>
            <ListItem disablePadding>
              <ListItemButton
                selected={selectedMenu === "Edit Nodes"}
                onClick={() => setSelectedMenu("Edit Nodes")}
              >
                <ListItemIcon>
                  <EditIcon />
                </ListItemIcon>
                <ListItemText primary="Edit Nodes" />
              </ListItemButton>
            </ListItem>
          </List>
          {/* Layers Form */}
          {selectedMenu === "Layers" && (
            <Box sx={{ p: 2 }}>
              <Typography variant="subtitle1" sx={{ mb: 2 }}>
                Add Layer
              </Typography>
              <TextField
                label="Layer label"
                value={newLabel}
                onChange={e => setNewLabel(e.target.value)}
                fullWidth
                size="small"
                sx={{ mb: 2 }}
              />
              <TextField
                select
                label="Layer Type"
                value={layerType}
                onChange={e => setLayerType(e.target.value)}
                fullWidth
                size="small"
                sx={{ mb: 2 }}
              >
                <MenuItem value="Linear">Linear</MenuItem>
                <MenuItem value="Convolutional">Convolutional</MenuItem>
                <MenuItem value="Flatten">Flatten</MenuItem>
              </TextField>
              {/* Show extra fields for Linear */}
              {layerType === "Linear" && (
                <>
                  <TextField
                    label="In Features"
                    value={inFeatures}
                    onChange={e => setInFeatures(e.target.value)}
                    type="number"
                    fullWidth
                    size="small"
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    label="Out Features"
                    value={outFeatures}
                    onChange={e => setOutFeatures(e.target.value)}
                    type="number"
                    fullWidth
                    size="small"
                    sx={{ mb: 2 }}
                  />
                </>
              )}
              <Button
                variant="contained"
                color="primary"
                fullWidth
                onClick={() => {
                  // Add new node to ReactFlow
                  const newId = `n${nodes.length + 1}`;
                  setNodes([
                    ...nodes,
                    {
                      id: newId,
                      position: { x: 100, y: 100 + nodes.length * 60 },
                      data: {
                        label: `${layerType}: ${newLabel || `Node ${nodes.length + 1}`}`,
                        ...(layerType === "Linear" && {
                          inFeatures,
                          outFeatures,
                        }),
                      },
                    },
                  ]);
                  setNewLabel("");
                  setInFeatures("");
                  setOutFeatures("");
                }}
              >
                Add Layer
              </Button>
            </Box>
          )}
        </div>
        {/* Export/Save Buttons */}
        <div>
          <Divider />
          <List>
            <ListItem disablePadding>
              <ListItemButton>
                <ListItemIcon>
                  <UploadIcon />
                </ListItemIcon>
                <ListItemText primary="Export" />
              </ListItemButton>
            </ListItem>
            <ListItem disablePadding>
              <ListItemButton>
                <ListItemIcon>
                  <SaveIcon />
                </ListItemIcon>
                <ListItemText primary="Save" />
              </ListItemButton>
            </ListItem>
          </List>
        </div>
      </Drawer>
      {/* Main Canvas Area */}
      <Main open={open}>
        <DrawerHeader />
        <Box sx={{ width: "100%", height: "80vh", background: "#f9fafb", borderRadius: 2, boxShadow: 1 }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            fitView
          />
        </Box>
      </Main>
    </Box>
  );
}
