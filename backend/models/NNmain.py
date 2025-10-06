from models.NNdatabase import NNDataBase

"""
    this is the file that should control of the proto NN
    Only the layers themselves are rescrited to Dicts, the Node and Structure can be classes
"""
DB = NNDataBase()

class NN():
    def __init__(self):
        self.Nodes = {} #
    pass

class Node():

    nnLayerID   = 0
    activatorID = 0
    tensorOpID  = 0
    def getNnLayerID():
        Node.nnLayerID += 1
        return Node.nnLayerID
    
    def getActivatorID():
        Node.activatorID += 1
        return Node.activatorID
    
    def getTensorOpID():
        Node.tensorOpID += 1
        return Node.tensorOpID
    
    IDfunc = {
        "nnLayer" : getNnLayerID,
        "activator" : getActivatorID,
        "tensorOp" : getTensorOpID
    }

    
    def __init__(self, layerType, layerName, parameters: dict, inLayers=None, outLayers=None):
        
        if layerType not in Node.IDfunc.keys(): raise ValueError
        if layerName not in DB.defaults[layerType].keys(): raise ValueError

        self.ID = layerType + "_" + str(Node.IDfunc[layerType]()) + "_" + layerType

        self.inLayers = inLayers
        self.outLayers = outLayers

        self.data = dict(DB.defaults[layerType][layerName]) # see JSON for details
        try:
            for k,v in parameters:
                if k not in self.data["parameters"]: raise KeyError #TODO: make custom error
                self.data["parameters"][k] = v
        except KeyError:
            raise KeyError


def getDefaultLayerParamters(layerType, layerName):
    """ 
        returns all default parameters for a given layerType, layerName
        \nsuch as: "activator, ReLU"
    """
    return DB.defaults[layerType][layerName]["parameters"]

def addLayer(layerType, layerName, parameters: dict, inLayers=None, outLayers=None):
    """
        adds layer to layerNetwork
    """
    pass

def removeLayer(layerID):
    """removes layer from network, #TODO: repair broken link, or break it permenetly"""
    pass

def modifyLayer(layerID, key, value):
    """ for parameters NOT input or outputs, given LayerID, key and value, adjusts value at that key """
    pass

def addLink(A_layerID, B_LayerID):
    """given layers A and B, CREATE a link between the output of A and input of B"""
    pass

def removeLink(A_layerID, B_LayerID):
    """given layers A and B, REMOVE a link between the output of A and input of B"""
    pass





    