import json

class NNDataBase():

    def __init__(self):

        self.defaults = {
            "nnLayer" : None,
            "tensorOp" : None,
            "activator" : None
        }

        with open("JsonLayers/NNLayers.json") as f:
            self.defaults["nnLayer"] = json.load(f)
        with open("JsonLayers/TensorOps.json") as f:
            self.defaults["tensorOp"] = json.load(f)
        with open("JsonLayers/Activators.json") as f:
            self.defaults["activator"] = json.load(f)

        # checks
        requiredParameters = [] #TODO: check each layer has the required params
        

    # def getNNLayer(self, layerName):
    #     try:
    #         return self.nnLayers[layerName]
    #     except KeyError:
    #         return None
        
    # def getTensorOp(self, tensorName):
    #     try:
    #         return self.tensorOps[tensorName]
    #     except KeyError:
    #         return None
    
    # def getActivator(self, activatorName):
    #     try:
    #         return self.activators[activatorName]
    #     except KeyError:
    #         return None
    
if __name__ == "__main__":
    pass