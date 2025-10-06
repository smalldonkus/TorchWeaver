from NNdatabase import NNDataBase

"""
    this is the file that should control of the proto NN
    Only the layers themselves are rescrited to Dicts, the Node and Structure can be classes
"""
DB = NNDataBase()

class Network():
    def __init__(self):
        self.nodes = []#

    def addNode(self, node):
        self.nodes.append(node)
    
    def removeNode(self, nodeID):
        for i in range(len(self.nodes)):
            if self.nodes[i].ID == nodeID:
                node = self.nodes.pop(i)
        # TODO: edit surrounding nodes

    def getNode(self, nodeID):
        for node in self.nodes:
            if node.ID == nodeID: 
                return node
        return None
    
    def modifyLayer(self, nodeID, key, value):

        """
            raises: ValueError
        """
        node: Node = self.getNode(nodeID)
        if node == None: raise ValueError #TODO: make custom Error
        try:
            node.setParameter(key, value)
        except ValueError:
            raise ValueError #TODO: custom Error

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
    
    def __init__(self, layerType, layerName, parameters: dict, inLayers: list[str], outLayers: list[str]):
        
        if layerType not in Node.IDfunc.keys(): raise ValueError
        if layerName not in DB.defaults[layerType].keys(): raise ValueError
        if (inLayers is not None) and len(inLayers) != 0:
            for n in inLayers:
                if not isinstance(n, str):
                    raise TypeError
        if (outLayers is not None) and len(outLayers) != 0:
            for n in outLayers:
                if not isinstance(n, str):
                    raise TypeError

        self.ID = layerType + "_" + str(Node.IDfunc[layerType]()) + "_" + layerName

        self.layerType = layerType
        self.layerName = layerName

        self.inLayers = inLayers
        self.outLayers = outLayers

        self.data = dict(DB.defaults[layerType][layerName]) # see JSON for details
        try:
            for k,v in parameters:
                if k not in self.data["parameters"]: raise KeyError #TODO: make custom error
                self.data["parameters"][k] = v
        except KeyError:
            raise KeyError
        
    def setParameter(self, paramKey, paramValue):
        if paramKey not in self.data["parameters"].keys(): raise ValueError #TODO: custom Error
        self.data["parameters"][paramKey] = paramValue
        return True
        
    def getParameters(self):
        return self.data["parameters"]
    
    def addInLayer(self, node):
        """
           arguments: Node = Node Object 
        """
        if (self.data["maxInputs"] is not None) and (len(self.inLayers) == self.data["maxInputs"]): 
            raise ValueError #TODO: find appropiate Error
        self.inLayers.append(node)

    def removeInLayer(self, nodeID):
        for i in range(len(self.inLayers)):
            if self.inLayers[i] == nodeID:
                self.inLayers.pop(i)
        if len(self.inLayers) < self.data["minInputs"]:
            raise ValueError #TODO: make custom layer
        
    def addOutLayer(self, node):
        """
            arguments: Node = Node Object 
        """
        self.outLayers.append(node)

    def removeOutLayer(self, nodeID):
        for i in range(len(self.outLayers)):
            if self.outLayers[i] == nodeID:
                self.outLayers.pop(i)

    def toDict(self):
        return {
            "layerType" : self.layerType,
            "layerName" : self.layerName,
            "inLayers"  : self.inLayers,
            "outLayers" : self.outLayers,
            "paramters" : self.parameters
        }


network = Network()

def getDefaultLayerParamters(layerType, layerName):
    """ 
        returns all default parameters for a given layerType, layerName
        \nsuch as: "activator, ReLU"
    """
    return DB.defaults[layerType][layerName]["parameters"]

def getLayerInfo(nodeID):
    """
        returns in the following format\n
        {
            "layerType"  : <layerType>,
            "layerType"  : <layerType>,
            "inLayers"   : [nodeID: node1, nodeID: node2 ...]
            "outLayers"  : [nodeID: node3, nodeID: node4 ...]
            "parameters" : {
                param1 : value1,
                param2 : value2,
                ...
                param_n: value_n,
            }
        }
        DO NOT MODIFY THIS OBJECT
    """
    n = network.getNode(nodeID)
    if n == None: raise ValueError
    return n.toDict()

def addLayer(layerType, layerName, parameters: dict, inLayers=None, outLayers=None):
    node = Node(
        layerType,
        layerName,
        parameters,
        inLayers,
        outLayers
    )
    network.addNode(node)

def removeLayer(layerID):
    """removes layer from network, #TODO: repair broken link, or break it permenetly"""
    pass

def modifyLayer(layerID, key, value):
    """ for parameters NOT input or outputs, given LayerID, key and value, adjusts value at that key """
    try:
        network.modifyLayer(layerID, key, value)
    except ValueError:
        raise ValueError
    

def addLink(A_layerID, B_LayerID):
    """given layers A and B, CREATE a link between the output of A and input of B"""
    pass

def removeLink(A_layerID, B_LayerID):
    """given layers A and B, REMOVE a link between the output of A and input of B"""
    pass

if __name__ == "__main__":
    pass    