export type NeuralNetworkInfo = {
  title: string;
  lastAccessed: string;
  image: string;
  Owner: string;
  Favourited: boolean;
};

let NeuralNetworks: NeuralNetworkInfo[] = [
    { title: "A Test Neural Network", lastAccessed: "10/10/1000", image: "/testnetwork.png" , Owner:"A", Favourited: true},
    { title: "B Test Neural Network", lastAccessed: "20/20/2000", image: "/testnetwork.png" , Owner:"A", Favourited: false},
    { title: "C Test Neural Network", lastAccessed: "30/30/3000", image: "/testnetwork.png" , Owner:"B", Favourited: true},
];

let VisibleNetworks: NeuralNetworkInfo[];

export const getNeuralNetworks = () => NeuralNetworks;

export const setNeuralNetworks = (NewNeuralNetworks: NeuralNetworkInfo[]) => {
  NeuralNetworks = NewNeuralNetworks;
};

export const setVisibleNetworks = (NewVisibleNetworks: NeuralNetworkInfo[]) => {
  VisibleNetworks = NewVisibleNetworks;
};

export const createNeuralNetwork = () => {
  //TODO: Write function to add to neuralnetworks
};

export const deleteNeuralNetwork = (NeuralNetwork: NeuralNetworkInfo) => { //TODO:CHECK THIS TYPE
  //TODO: write function to remove from neuralnetworks
};