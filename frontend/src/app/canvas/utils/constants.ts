export const drawerWidth = 240;

// export const initialNodes = [
//   {
//     id: "n1",
//     position: { x: 0, y: 0 },
//     data: {
//       label: "Linear: Node 1",
//       operationType: "Layer",
//       type: "Linear",
//       parameters: { in_features: "2", out_features: "1" },
//     },
//   },
//   {
//     id: "n2",
//     position: { x: 0, y: 100 },
//     data: {
//       label: "Linear: Node 2",
//       operationType: "Layer",
//       type: "Linear",
//       parameters: { in_features: "4", out_features: "2" },
//     },
//   },
//   {
//     id: "n3",
//     position: { x: 0, y: 200 },
//     data: {
//       label: "Linear: Node 3",
//       operationType: "Layer",
//       type: "Linear",
//       parameters: { in_features: "8", out_features: "4" },
//     },
//   },
// ];

// export const initialEdges = [{ id: "n1-n2", source: "n1", target: "n2" }];

// AlexNet predefined network
export const initialNodes = [
  //Input node
  // {
  //   id: "input",
  //   position: { x: 100, y: 50 },
  //   data: {
  //     label: "Input: 3x224x224",
  //     operationType: "Input",
  //     parameters: {
  //       shapeType: "3D",
  //       dims: [3, 224, 224],
  //     },
  //   },
  // },
  // // Convolutional layers
  // {
  //   id: "conv1",
  //   position: { x: 100, y: 150 },
  //   data: {
  //     label: "Conv2d: conv1",
  //     operationType: "Layer",
  //     type: "Conv2d",
  //     parameters: {
  //       in_channels: 3,
  //       out_channels: 96,
  //       kernel_size: 11,
  //       stride: 4,
  //       padding: 2,
  //       bias: true
  //     },
  //   },
  // },
  // {
  //   id: "relu1",
  //   position: { x: 100, y: 250 },
  //   data: {
  //     label: "ReLU: relu1",
  //     operationType: "Activator",
  //     type: "ReLU",
  //     parameters: {},
  //   },
  // },
  // {
  //   id: "lrn1",
  //   position: { x: 100, y: 350 },
  //   data: {
  //     label: "LocalResponseNorm: lrn1",
  //     operationType: "Layer",
  //     type: "LocalResponseNorm",
  //     parameters: {
  //       size: 5,
  //       alpha: 0.0001,
  //       beta: 0.75,
  //       k: 2.0
  //     },
  //   },
  // },
  // {
  //   id: "pool1",
  //   position: { x: 100, y: 450 },
  //   data: {
  //     label: "MaxPool2d: pool1",
  //     operationType: "Layer",
  //     type: "MaxPool2d",
  //     parameters: {
  //       kernel_size: 3,
  //       stride: 2
  //     },
  //   },
  // },
  // {
  //   id: "conv2",
  //   position: { x: 300, y: 150 },
  //   data: {
  //     label: "Conv2d: conv2",
  //     operationType: "Layer",
  //     type: "Conv2d",
  //     parameters: {
  //       in_channels: 96,
  //       out_channels: 256,
  //       kernel_size: 5,
  //       stride: 1,
  //       padding: 2,
  //       bias: true
  //     },
  //   },
  // },
  // {
  //   id: "relu2",
  //   position: { x: 300, y: 250 },
  //   data: {
  //     label: "ReLU: relu2",
  //     operationType: "Activator",
  //     type: "ReLU",
  //     parameters: {},
  //   },
  // },
  // {
  //   id: "lrn2",
  //   position: { x: 300, y: 350 },
  //   data: {
  //     label: "LocalResponseNorm: lrn2",
  //     operationType: "Layer",
  //     type: "LocalResponseNorm",
  //     parameters: {
  //       size: 5,
  //       alpha: 0.0001,
  //       beta: 0.75,
  //       k: 2.0
  //     },
  //   },
  // },
  // {
  //   id: "pool2",
  //   position: { x: 300, y: 450 },
  //   data: {
  //     label: "MaxPool2d: pool2",
  //     operationType: "Layer",
  //     type: "MaxPool2d",
  //     parameters: {
  //       kernel_size: 3,
  //       stride: 2
  //     },
  //   },
  // },
  // {
  //   id: "conv3",
  //   position: { x: 500, y: 150 },
  //   data: {
  //     label: "Conv2d: conv3",
  //     operationType: "Layer",
  //     type: "Conv2d",
  //     parameters: {
  //       in_channels: 256,
  //       out_channels: 384,
  //       kernel_size: 3,
  //       stride: 1,
  //       padding: 1,
  //       bias: true
  //     },
  //   },
  // },
  // {
  //   id: "relu3",
  //   position: { x: 500, y: 250 },
  //   data: {
  //     label: "ReLU: relu3",
  //     operationType: "Activator",
  //     type: "ReLU",
  //     parameters: {},
  //   },
  // },
  // {
  //   id: "conv4",
  //   position: { x: 700, y: 150 },
  //   data: {
  //     label: "Conv2d: conv4",
  //     operationType: "Layer",
  //     type: "Conv2d",
  //     parameters: {
  //       in_channels: 384,
  //       out_channels: 384,
  //       kernel_size: 3,
  //       stride: 1,
  //       padding: 1,
  //       bias: true
  //     },
  //   },
  // },
  // {
  //   id: "relu4",
  //   position: { x: 700, y: 250 },
  //   data: {
  //     label: "ReLU: relu4",
  //     operationType: "Activator",
  //     type: "ReLU",
  //     parameters: {},
  //   },
  // },
  // {
  //   id: "conv5",
  //   position: { x: 900, y: 150 },
  //   data: {
  //     label: "Conv2d: conv5",
  //     operationType: "Layer",
  //     type: "Conv2d",
  //     parameters: {
  //       in_channels: 384,
  //       out_channels: 256,
  //       kernel_size: 3,
  //       stride: 1,
  //       padding: 1,
  //       bias: true
  //     },
  //   },
  // },
  // {
  //   id: "relu5",
  //   position: { x: 900, y: 250 },
  //   data: {
  //     label: "ReLU: relu5",
  //     operationType: "Activator",
  //     type: "ReLU",
  //     parameters: {},
  //   },
  // },
  // {
  //   id: "pool5",
  //   position: { x: 900, y: 350 },
  //   data: {
  //     label: "MaxPool2d: pool5",
  //     operationType: "Layer",
  //     type: "MaxPool2d",
  //     parameters: {
  //       kernel_size: 3,
  //       stride: 2
  //     },
  //   },
  // },
  // {
  //   id: "flatten1",
  //   position: { x: 900, y: 450 },
  //   data: {
  //     label: "Flatten: flatten1",
  //     operationType: "Layer",
  //     type: "Flatten",
  //     parameters: {},
  //   },
  // },
  // {
  //   id: "dropout1",
  //   position: { x: 900, y: 550 },
  //   data: {
  //     label: "Dropout: dropout1",
  //     operationType: "Layer",
  //     type: "Dropout",
  //     parameters: {
  //       p: 0.5
  //     },
  //   },
  // },
  // {
  //   id: "fc1",
  //   position: { x: 1100, y: 150 },
  //   data: {
  //     label: "Linear: fc1",
  //     operationType: "Layer",
  //     type: "Linear",
  //     parameters: {
  //       in_features: 9216,
  //       out_features: 4096,
  //       bias: true
  //     },
  //   },
  // },
  // {
  //   id: "relu6",
  //   position: { x: 1100, y: 250 },
  //   data: {
  //     label: "ReLU: relu6",
  //     operationType: "Activator",
  //     type: "ReLU",
  //     parameters: {},
  //   },
  // },
  // {
  //   id: "dropout2",
  //   position: { x: 1100, y: 350 },
  //   data: {
  //     label: "Dropout: dropout2",
  //     operationType: "Layer",
  //     type: "Dropout",
  //     parameters: {
  //       p: 0.5
  //     },
  //   },
  // },
  // {
  //   id: "fc2",
  //   position: { x: 1300, y: 150 },
  //   data: {
  //     label: "Linear: fc2",
  //     operationType: "Layer",
  //     type: "Linear",
  //     parameters: {
  //       in_features: 4096,
  //       out_features: 4096,
  //       bias: true
  //     },
  //   },
  // },
  // {
  //   id: "relu7",
  //   position: { x: 1300, y: 250 },
  //   data: {
  //     label: "ReLU: relu7",
  //     operationType: "Activator",
  //     type: "ReLU",
  //     parameters: {},
  //   },
  // },
  // {
  //   id: "fc3",
  //   position: { x: 1500, y: 150 },
  //   data: {
  //     label: "Linear: fc3",
  //     operationType: "Layer",
  //     type: "Linear",
  //     parameters: {
  //       in_features: 4096,
  //       out_features: 1000,
  //       bias: true
  //     },
  //   },
  // },
];

export const initialEdges = [
  // { id: "input-conv1", source: "input", target: "conv1" },
  // { id: "conv1-relu1", source: "conv1", target: "relu1" },
  // { id: "relu1-lrn1", source: "relu1", target: "lrn1" },
  // { id: "lrn1-pool1", source: "lrn1", target: "pool1" },
  // { id: "pool1-conv2", source: "pool1", target: "conv2" },
  // { id: "conv2-relu2", source: "conv2", target: "relu2" },
  // { id: "relu2-lrn2", source: "relu2", target: "lrn2" },
  // { id: "lrn2-pool2", source: "lrn2", target: "pool2" },
  // { id: "pool2-conv3", source: "pool2", target: "conv3" },
  // { id: "conv3-relu3", source: "conv3", target: "relu3" },
  // { id: "relu3-conv4", source: "relu3", target: "conv4" },
  // { id: "conv4-relu4", source: "conv4", target: "relu4" },
  // { id: "relu4-conv5", source: "relu4", target: "conv5" },
  // { id: "conv5-relu5", source: "conv5", target: "relu5" },
  // { id: "relu5-pool5", source: "relu5", target: "pool5" },
  // { id: "pool5-flatten1", source: "pool5", target: "flatten1" },
  // { id: "flatten1-dropout1", source: "flatten1", target: "dropout1" },
  // { id: "dropout1-fc1", source: "dropout1", target: "fc1" },
  // { id: "fc1-relu6", source: "fc1", target: "relu6" },
  // { id: "relu6-dropout2", source: "relu6", target: "dropout2" },
  // { id: "dropout2-fc2", source: "dropout2", target: "fc2" },
  // { id: "fc2-relu7", source: "fc2", target: "relu7" },
  // { id: "relu7-fc3", source: "relu7", target: "fc3" },
];
