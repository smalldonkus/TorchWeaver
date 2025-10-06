from NNdatabase import NNDataBase

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

    ID = 0
    def getID():
        Node.ID += 1
        return Node.ID
    
    
    def __init__(self, layerType, layerName, inLayers, outLayers, arguments: dict):
        default = None
        try:
            match layerType.lower():
                case "activation":
                    default = DB.getActivator(layerName)
                case "nnlayer":
                    default = DB.getNNLayer(layerName)
                case "tensoroperator":
                    default = DB.getTensorOp(layerName)
        except KeyError:
            raise KeyError #TODO:
        
        self.ID = Node.getID()

        self.inLayers = inLayers
        self.outLayers = outLayers

        self.operation = default # contains the relevant data from the operation

# gets the relevant layer information
def getInfoOnLayer():
    pass    



    