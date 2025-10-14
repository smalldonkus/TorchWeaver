export const drawerWidth = 240;

export const initialNodes = [
  {
    id: "n1",
    position: { x: 0, y: 0 },
    data: {
      label: "Linear: Node 1",
      operationType: "Layer",
      type: "Linear",
      parameters: { in_features: "2", out_features: "1" },
    },
  },
  {
    id: "n2",
    position: { x: 0, y: 100 },
    data: {
      label: "Linear: Node 2",
      operationType: "Layer",
      type: "Linear",
      parameters: { in_features: "4", out_features: "2" },
    },
  },
  {
    id: "n3",
    position: { x: 0, y: 200 },
    data: {
      label: "Linear: Node 3",
      operationType: "Layer",
      type: "Linear",
      parameters: { in_features: "8", out_features: "4" },
    },
  },
];

export const initialEdges = [{ id: "n1-n2", source: "n1", target: "n2" }];
