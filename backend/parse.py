
from NNdatabase import NNDataBase


CRUDE_REPORT = False

class ParseError:

    def __init__(self, desc, nodeIDs:list[str]=None):
        self.desc = desc
        self.nodes = None if nodeIDs is None else nodeIDs

    def report(self):
        return self.desc

def find(nodesList, nodeId):
    query = [n for n in nodesList if n["id"] == nodeId]
    return None if len(query) == 0 else query[0]

class Graph:

    def __init__(self, nodesList):

        # ** TARJAN ** #
        self.V = []
        self.E = []
        self.indices = {}
        self.lowlink = {}
        self.onStack = {}

        self.index = 0
        self.S = []

        self.SSCs = []
        self.hasRunTarjan = False
        # ** TARJAN ** #

        # ** MATCHING INCOMING OUTGOING ** #
        self.inputChannels = {}
        self.outputChannels = {}
        self.isOutput = {}
        # self.canInherit = {}
        # ** MATCHING INCOMING OUTGOING ** #

        # populate graph
        for n in nodesList:
            self.V.append(n["id"])
            for c in n["children"]:
                self.E.append([n["id"], c])

            # ** TARJAN ** #
            self.indices[n["id"]] = None
            self.lowlink[n["id"]] = None
            self.onStack[n["id"]] = False
            # ** TARJAN ** #
            
            # ** MATCHING INCOMING OUTGOING ** #
            isOutput = n["data"]["operationType"] == "Output"
            self.inputChannels[n["id"]]  = None if isOutput else n["data"]["inputChannels"]
            self.outputChannels[n["id"]] = None if isOutput else n["data"]["outputChannels"]
            # Check if node can inherit from parent (skip validation for these nodes)
            # self.canInherit[n["id"]] = n["data"].get("can_inherit_from_parent", False) or n["data"].get("parameters", {}).get("inherit_from_parent", False)
            # ** MATCHING INCOMING OUTGOING ** #

            self.isOutput[n["id"]] = True if isOutput else False

    # returing list in format [(parentInError, childInError), ...]
    def hasMatchingInAndOut(self):
        rtn = []
        for v in self.V:

            if self.inputChannels[v] is None: continue

            for e in self.E:

                if e[0] == v:
                    w = e[1]
                    if self.inputChannels[w] is None: continue
                    
                    # Skip validation if child node can inherit (it will auto-match parent)
                    # if self.canInherit.get(w, False): continue

                    if self.outputChannels[v] != self.inputChannels[w]:
                        rtn.append((v,w))
        return rtn

    def tarjan(self):
        # puesdoCode: https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
        assert len(self.S) == 0 and self.index == 0 # needs to be rePop'd everytime

        for v in self.V:
            if self.indices[v] is None:
                self.strongConnect(v)

        self.hasRunTarjan = True

    def getInError(self):
        assert self.hasRunTarjan

        inErr = []
        for ssc in self.SSCs:
            if len(ssc) > 1:
                for n in ssc:
                    inErr.append(n)
        return None if len(inErr) == 0 else inErr

    def strongConnect(self, v):
        
        self.indices[v] = self.index
        self.lowlink[v] = self.index
        self.index += 1
        self.S.append(v)
        self.onStack[v] = True

        for w in [e[1] for e in self.E if e[0] == v]:
            
            if self.indices[w] is None:
                self.strongConnect(w) #recurse
                self.lowlink[v] = min(self.lowlink[v], self.lowlink[w])
            elif self.onStack[w]:
                self.lowlink[v] = min(self.lowlink[v], self.indices[w])

        if self.lowlink[v] == self.indices[v]:
            cond = True
            SCC = []
            while cond:
                w = self.S.pop()
                self.onStack[w] = False
                SCC.append(w)
                cond = (w != v)
            self.SSCs.append(SCC)
    
    def isPathFromInputToOutput(self, origin: str):
        
        q = [origin]
        vst = [] # id list

        while len(q) != 0:

            n = q.pop()

            if n in vst: 
                continue
            vst.append(n)

            # assumes children list contains only ID's

            for e in self.E:
                
                if e[0] != n: continue

                c = e[1]
                if self.isOutput[c]: 
                    return True                
                    
                q.append(c)

        return False

def parse(nodesList):

    # errors list
    """
        every error needs to have a unique ID and description, as errors are fixed, the ID's
        need to change, storing in order in a queue is probably the best way.
    """
    DB = NNDataBase() # this object is shit, fixed below
    
    errors = []

    inputs = [n for n in nodesList if (n["data"]["operationType"]).lower() == "input"]
    # checks for input existing
    if len(inputs) == 0:
        errors.append(ParseError("Please define an Input", nodeIDs=None))

    # checks for input's Having a parent
    for n in inputs:
        if len(n.get("parents", [])) != 0:
            errors.append(ParseError("Inputs cannot have a parent", nodeIDs=[n["id"]]))

    outputs = [n for n in nodesList if (n["data"]["operationType"]).lower() == "output"]
    if len(outputs) == 0:
        errors.append(ParseError("Please define an Output", nodeIDs=None))
    elif len(outputs) > 1:
        errors.append(ParseError("Too many outputs, only one is permitted", nodeIDs=[n["id"] for n in outputs]))
    
    for n in outputs:
        if len(n.get("children", [])) != 0:
            errors.append(ParseError("Outputs cannot have children", nodeIDs=[n["id"]]))
        if len(n.get("parents", [])) == 0:
            errors.append(ParseError("Outputs need an a parent node", nodeIDs=[n["id"]]))
        if len(n.get("parents", [])) > 1:
            errors.append(ParseError(f"Output requires only one parent, currently has {len(n.get("parents", []))}", nodeIDs=[n["id"]]))

    # checks for maxInputs and minInputs being obeyed
    for n in nodesList:
        if n["data"]["operationType"] == "Input": continue # checked elsewhere
        if n["data"]["operationType"] == "Output": continue # has no default

        # Find the default configuration for this node type using hierarchical lookup
        node_type = n["data"].get("type")
        if not node_type:
            errors.append(ParseError(f"Node {n['id']} missing type field", nodeIDs=[n["id"]]))
            continue
            
        dflt = DB.find_definition(node_type)
        
        if dflt is None:
            errors.append(ParseError(f"No default configuration found for node type {node_type}", nodeIDs=[n["id"]]))
            continue
            
        # Check parseCheck constraints if they exist
        parse_check = dflt.get("parseCheck", {})
        min_inputs = parse_check.get("minInputs", 0)
        max_inputs = parse_check.get("maxInputs", float('inf'))
        
        parent_count = len(n.get("parents", []))
        if parent_count < int(min_inputs):
            errors.append(ParseError(f"{dflt['type']} requires at least {min_inputs} parent{'s' if min_inputs > 1 else ''}, currently has {parent_count}", nodeIDs=[n["id"]]))
        if parent_count > int(max_inputs):
            errors.append(ParseError(f"{dflt['type']} requires less or equal to {max_inputs} parent{'s' if max_inputs > 1 else ''}, currently has {parent_count}", nodeIDs=[n["id"]]))
  
    # check for cycles.
    nG = Graph(nodesList)
    nG.tarjan()
    inErr = nG.getInError()
    if inErr is not None:
        errors.append(ParseError(f"This is node is part of a cycle", nodeIDs=inErr))

    # get all inputs for graph
    inputs = [n["id"] for n in nodesList if n["data"]["operationType"] == "Input"]
    # check for path from inputs to output
    pathCheck = [(nG.isPathFromInputToOutput(i), i) for i in inputs]
    for path in pathCheck:
        if not path[0]: errors.append(ParseError(f"This input has no path to an output", nodeIDs=[path[1]]))

    # check for matching inputChannels to outputChannels
    # inErr = nG.hasMatchingInAndOut()
    # if len(inErr) != 0:
    #     for err in inErr:
    #         errors.append(ParseError(f"Output dimensions of parent do not match this node's input dimensions, nodes such as activators/tensor operations, can inherit dimensions. You may need to go futher up the graph to fix this error.",
    #                       nodeIDs=[err[1]]))
    #         errors.append(ParseError(f"The output dimensions of this node do match the input dimensions of its children, nodes such as activators/tensor operations, can inherit dimensions. You may need to go futher up the graph to fix this error.",
    #                       nodeIDs=[err[0]]))

    if CRUDE_REPORT:
        for i, e in enumerate(errors):
            print(e.report())

    return [] if len(errors) == 0 else [
        {
            "id": i + 1,
            "errorMsg" : e.report(),
            "flaggedNodes" : e.nodes
        }
        for i, e in enumerate(errors)
    ]

if __name__ == "__main__":
    # test taken from here: https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
    
    """ ************ Tarjan Tests Begin ************ """
    
    NL1 = [
        {"id" : "1", "children" : ["2"]},
        {"id" : "2", "children" : ["3"]},
        {"id" : "3", "children" : ["1"]},
        {"id" : "4", "children" : ["2", "3", "5"]},
        {"id" : "5", "children" : ["4", "6"]},
        {"id" : "6", "children" : ["3", "7"]},
        {"id" : "7", "children" : ["6"]},
        {"id" : "8", "children" : ["5", "7", "8"]}
    ]
    G1 = Graph(NL1)
    result = G1.tarjan()
    assert G1.SSCs == [['3', '2', '1'], ['7', '6'], ['5', '4'], ['8']]

    NL2 = [
        {"id" : "A", "children" : ["B"]},
        {"id" : "B", "children" : ["C", "D"]},
        {"id" : "C", "children" : ["E"]},
        {"id" : "D", "children" : ["F"]},
        {"id" : "E", "children" : ["C"]},
        {"id" : "F", "children" : ["D", "G"]},
        {"id" : "G", "children" : []},
    ]
    G2 = Graph(NL2)
    G2.tarjan()
    assert G2.SSCs == [['E', 'C'], ['G'], ['F', 'D'], ['B'], ['A']]

    """ ************ Tarjan Tests End ************ """
