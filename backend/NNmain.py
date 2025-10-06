from NNdatabase import NNDataBase

"""
    this is the file that should control of the proto NN
    Only the layers themselves are rescrited to Dicts, the Node and Structure can be classes
"""
DB = NNDataBase()

class Network():
    def __init__(self):
        self.nodes = []#

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
        
    def getLayerInfo(self, nodeID):
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
        n = self.getNode(nodeID)
        if n == None: raise ValueError
        return n.toDict()

    def addLayer(self, layerType, layerName, parameters: dict, inLayers=None, outLayers=None):
        node = Node(
            layerType,
            layerName,
            parameters,
            inLayers,
            outLayers
        )
        self.nodes.append(node)

    def removeLayer(self, layerID):
        """removes layer from network, #TODO: repair broken link, or break it permenetly"""
        n: Node = self.getNode(layerID)
        if n is None: raise ValueError #TODO: make custom Error
        # TODO: review functionality
        # clear inlayers reference and outlayers reference of this node
        for nInID in n.inLayers:
            nIn: Node = self.getNode(nInID)
            nIn.outLayers = [e for e in nIn.outLayers if e.ID != n.ID]
        for nOutID in n.outLayers:
            nOut: Node = self.getNode(nOutID)
            nOut.inLayers = [e for e in nOut.outLayers if e.ID != n.ID]

    def addLink(self, A_layerID, B_LayerID):
        """given layers A and B, CREATE a link between the output of A and input of B"""
        a: Node = self.getNode(A_layerID)
        b: Node = self.getNode(B_LayerID)
        if a is None or b is None: raise ValueError #TODO: make custom error

        a.addOutLayer(b.ID)
        b.addInLayer(a.ID)        

    def removeLink(self, A_layerID, B_LayerID):
        """given layers A and B, REMOVE a link between the output of A and input of B"""
        a: Node = self.getNode(A_layerID)
        b: Node = self.getNode(B_LayerID)
        if a is None or b is None: raise ValueError #TODO: make custom error

        a.removeOutLayer(b.ID)
        b.removeInLayer(a.ID)      
        pass

class Node():

    nnLayerID   = 0 #TODO: edit for persistance
    activatorID = 0 #TODO: edit for persistance
    tensorOpID  = 0 #TODO: edit for persistance
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

        self.inLayers : list[str] = inLayers
        self.outLayers: list[str] = outLayers

        self.data = dict(DB.defaults[layerType][layerName]) # see JSON for details
        try:
            for k,v in parameters:
                if k not in self.data["parameters"]: raise KeyError #TODO: make custom error
                self.data["parameters"][k] = v
        except KeyError:
            raise KeyError
        
    def setParameter(self, paramKey, paramValue):
        if paramKey not in self.data["parameters"].keys(): raise ValueError #TODO: make custom error
        self.data["parameters"][paramKey] = paramValue
        return True
        
    def getParameters(self):
        return self.data["parameters"]
    
    def addInLayer(self, nodeID: str):
       
        if not isinstance(nodeID, str): 
            raise TypeError
        if (self.data["maxInputs"] is not None) and (len(self.inLayers) == self.data["maxInputs"]): 
            raise ValueError #TODO: make custom error
        if nodeID in self.outLayers:
            raise RuntimeError #TODO: make custom error
        self.inLayers.append(nodeID)

    def removeInLayer(self, nodeID: str):
        for i in range(len(self.inLayers)):
            if self.inLayers[i] == nodeID:
                self.inLayers.pop(i)
                return
        if len(self.inLayers) < self.data["minInputs"]:
            raise ValueError #TODO: make custom layer
        
    def addOutLayer(self, nodeID: str):
        
        if not isinstance(nodeID, str): 
            raise TypeError
        if nodeID in self.outLayers:
            raise RuntimeError #TODO: make custom error
        self.outLayers.append(nodeID)

    def removeOutLayer(self, nodeID: str):
        for i in range(len(self.outLayers)):
            if self.outLayers[i] == nodeID:
                self.outLayers.pop(i)
                return

    def toDict(self):
        return {
            "layerType" : self.layerType,
            "layerName" : self.layerName,
            "inLayers"  : self.inLayers,
            "outLayers" : self.outLayers,
            "paramters" : self.parameters
        }

def getDefaultLayerParamters(layerType, layerName):
    """ 
        returns all default parameters for a given layerType, layerName
        \nsuch as: "activator, ReLU"
    """
    return DB.defaults[layerType][layerName]["parameters"]

if __name__ == "__main__":
    nn = Network() 