from NNdatabase import NNDataBase

"""
    this is the file that should control of the proto NN
    Only the layers themselves are rescrited to Dicts, the Node and Structure can be classes
"""
DB = NNDataBase()

class Network():

    def __init__(self):
        
        self.input = InNode() # acts as head
        assert self.input.outLayers.__len__() == 0, f"{self.input.outLayers}" # avoids odd bug
        self.output = OutNode()
        assert self.output.inLayers.__len__() == 0, f"{self.output.inLayers}" # avoids odd bug
        self.nodes = [self.input, self.output]
        self.addLink(self.input.ID, self.output.ID)

    def popNode(self, nodeID):
        for i in range(len(self.nodes)):
            if self.nodes[i].ID == nodeID:
                return self.nodes.pop(i)

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
        
    def getLayerInfo(self, nodeID) -> dict: 
        """
            returns in the following format\n
            {
                "layerType"  : <layerType>,
                "layerType"  : <layerType>,
                "inLayers"   : [nodeID, nodeID ... nodeID]
                "outLayers"  : [nodeID, nodeID ... nodeID]
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

    def addLayer(self, layerType, layerName, parameters: dict=None, inLayers=None, outLayers=None):
        node = Node(
            layerType,
            layerName,
            parameters,
            inLayers,
            outLayers
        )
        self.nodes.append(node)
        return node.ID

    def removeLayer(self, layerID):
        """removes layer from network, removes all links i.e creates gap in network #TODO: repair broken link, or break it permenetly"""
        n: Node = self.popNode(layerID)
        if n is None: raise ValueError #TODO: make custom Error
        # TODO: review functionality
        # clear inlayers reference and outlayers reference of this node
        for nInID in n.inLayers:
            nIn: Node = self.getNode(nInID)
            nIn.outLayers = [e for e in nIn.outLayers if e != n.ID]
        
        for nOutID in n.outLayers:
            nOut: Node = self.getNode(nOutID)
            nOut.inLayers = [e for e in nOut.outLayers if e != n.ID]

    def addLink(self, A_layerID, B_LayerID):
        """given layers A and B, CREATE a link between the output of A and input of B"""
        a: Node = self.getNode(A_layerID)
        b: Node = self.getNode(B_LayerID)
        if a is None or b is None: raise ValueError #TODO: make custom error

        a.addOutLayer(b.ID)
        b.addInLayer(a.ID)        
    
    def addInBetweenLink(self, A_layerID, B_LayerID, C_LayerID): #UNTESTED: TODO: flesh out
        """
            creates a link between A-B-C, with;
            A being the origin (leftmost)
            B being the centre,
            C being the end (rightmost)
        """
        self.addLink(A_layerID, B_LayerID)
        self.addLink(B_LayerID, C_LayerID)

    def removeLink(self, A_layerID, B_LayerID):
        """given layers A and B, REMOVE a link between the output of A and input of B"""
        a: Node = self.getNode(A_layerID)
        b: Node = self.getNode(B_LayerID)
        if a is None or b is None: raise ValueError #TODO: make custom error

        a.removeOutLayer(b.ID)
        b.removeInLayer(a.ID)      
        pass

    def __str__(self):
        """ prints the network from the top down """
        # TODO: add BFS like algo, sort out problems relating to broken links
        # assumes sequential list
        s = ""
        currN = self.input
        while currN != None:
            s += f"{currN.ID}{"\n" if currN.ID != "OUT" else ""}"
            currN = None if len(currN.outLayers) == 0 else self.getNode(currN.outLayers[0])
        return s

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
        
    def resetIDs(secrectPassword):
        assert secrectPassword == "plsNo"
        Node.nnLayerID, Node.activatorID, Node.tensorOpID = 0, 0, 0
        # only for testing

    IDfunc = {
        "nnLayer" : getNnLayerID,
        "activator" : getActivatorID,
        "tensorOP" : getTensorOpID
    }
    
    def __init__(self, layerType, layerName, parameters: dict, inLayers: list[str]=None, outLayers: list[str]=None):
        
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

        self.inLayers : list[str] = [] if inLayers is None else inLayers
        self.outLayers: list[str] = [] if outLayers is None else outLayers

        self.data = dict(DB.defaults[layerType][layerName]) # see JSON for details
        if parameters is not None and len(parameters) != 0:
            try:
                for k,v in parameters.items():
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
            raise RuntimeError(f"{self.outLayers} includes {nodeID}") #TODO: make custom error
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
            "parameters" : self.data["parameters"]
        }
    
    def __str__(self):
        return f"{self.ID}"

class InNode(Node):

    def __init__(self, outLayers: list[str] = None):
        
        self.ID = "IN"

        self.inLayers : list[str] = []
        self.outLayers: list[str] = [] if outLayers is None else outLayers

        self.layerType = "TorchWeaver"
        self.layerName = "IN"

        self.data = {
            "library" : None,
            "className" : None,

            "inputParamName" : None,
            "maxInputs" : 0,
            "minInputs" : 0,
            "isFunction" : False,

            "parameters" : {}
        }
    def setParameter(self, paramKey, paramValue):
        raise ReferenceError # TODO: make custom error
        
    def addInLayer(self, nodeID: str):
        raise ReferenceError # TODO: make custom error

    def removeInLayer(self, nodeID: str):
        raise ReferenceError # TODO: make custom error

class OutNode(Node):

    def __init__(self, inLayers: list[str] = None):
        
        self.ID = "OUT"

        self.inLayers : list[str] = [] if inLayers is None else inLayers
        self.outLayers: list[str] = []

        self.layerType = "TorchWeaver"
        self.layerName = "OUT"

        self.data = {
            "library" : None,
            "className" : None,

            "inputParamName" : None,
            "maxInputs" : None,
            "minInputs" : 1,
            "isFunction" : False,

            "parameters" : {}
        }
    def setParameter(self, paramKey, paramValue):
        raise ReferenceError # TODO: make custom error
        
    def addOutLayer(self, nodeID: str):
        raise ReferenceError # TODO: make custom error

    def removeOutLayer(self, nodeID: str):
        raise ReferenceError # TODO: make custom error  

def getDefaultLayerParamters(layerType, layerName):
    """ 
        returns all default parameters for a given layerType, layerName
        \nsuch as: "activator, ReLU"
    """
    return DB.defaults[layerType][layerName]["parameters"]

if __name__ == "__main__":
    
    #test 1
    nn = Network()
    assert nn.__str__() == "IN\nOUT"
    Node.resetIDs("plsNo")
    del nn

    #test 2: add link, remove link
    nn = Network()
    node1_ID = nn.addLayer("nnLayer", "Linear")
    nn.addLink(nn.input.ID, node1_ID)
    nn.addLink(node1_ID, nn.output.ID)
    nn.removeLink(nn.input.ID, nn.output.ID)
    assert nn.__str__() == "IN\nnnLayer_1_Linear\nOUT"
    Node.resetIDs("plsNo")
    del nn

    # test 3: remove layer
    nn = Network()
    node1_ID = nn.addLayer("nnLayer", "Linear")
    nn.addLink(nn.input.ID, node1_ID)
    nn.addLink(node1_ID, nn.output.ID)
    nn.removeLayer(node1_ID)
    assert nn.__str__() == "IN\nOUT"
    Node.resetIDs("plsNo")
    del nn

    # test 4: modify parameters, VER1 adding at init
    # tests: addLayer, addLink, getLayerInfo
    nn = Network()
    node1_ID = nn.addLayer("nnLayer", "Linear", parameters={"in_features" : 256, "out_features" : 128})
    nn.addLink(nn.input.ID, node1_ID)
    nn.addLink(node1_ID, nn.output.ID)
    nn.removeLink(nn.input.ID, nn.output.ID)
    dct = nn.getLayerInfo(node1_ID)
    assert dct["parameters"]["in_features"] == 256 and dct["parameters"]["out_features"] == 128
    assert nn.__str__() == "IN\nnnLayer_1_Linear\nOUT", nn.__str__()
    Node.resetIDs("plsNo")
    del nn

    nn = Network()
    node1_ID = nn.addLayer("nnLayer", "Linear")
    nn.addLink(nn.input.ID, node1_ID)
    nn.addLink(node1_ID, nn.output.ID)
    nn.removeLink(nn.input.ID, nn.output.ID)
    nn.modifyLayer(node1_ID, "in_features", 256)
    nn.modifyLayer(node1_ID, "out_features", 128)
    dct = nn.getLayerInfo(node1_ID)
    assert dct["parameters"]["in_features"] == 256 and dct["parameters"]["out_features"] == 128
    assert nn.__str__() == "IN\nnnLayer_1_Linear\nOUT", nn.__str__()
    Node.resetIDs("plsNo")
    del nn