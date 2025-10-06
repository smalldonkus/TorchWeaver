import json

class NNDataBase():

    def __init__(self):
        with open("JsonLayers/NNLayers.json") as f:
            self.nnLayers = json.load(f)
        with open("JsonLayers/TensorOps.json") as f:
            self.tensorOps = json.load(f)
        with open("JsonLayers/Activators.json") as f:
            self.activators = json.load(f)

    def getNNLayer(self, layerName):
        try:
            return self.nnLayers[layerName]
        except KeyError:
            return None
        
    def getTensorOp(self, tensorName):
        try:
            return self.tensorOps[tensorName]
        except KeyError:
            return None
    
    def getActivator(self, activatorName):
        try:
            return self.activators[activatorName]
        except KeyError:
            return None
    
if __name__ == "__main__":
    pass