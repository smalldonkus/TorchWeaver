
from NNdatabase import NNDataBase
from collections import deque

CRUDE_REPORT = True
DEBUG = True

class ParseError:

    def __init__(self, desc, nodeIDs:list[str]=None):
        self.desc = desc
        self.nodes = None if nodeIDs is None else nodeIDs

    def report(self):
        return self.desc

def find(nodesList, nodeId):
    query = [n for n in nodesList if n["id"] == nodeId]
    return None if len(query) == 0 else query[0]

def dfs(nodesList, origin):

    q = [origin]
    v = [] # id list

    while len(q) != 0:

        n = q.pop()

        assert isinstance(n, dict)

        if n["id"] in v: 
            continue
        v.append(n["id"])

        # assumes children list contains only ID's

        for c in n["children"]:
            
            child = find(nodesList, c)
            if child is None: 
                raise RuntimeError(f"DFS: couldn't find {c} in nodes list for node {n["id"]}")
            
            assert isinstance(child, dict)
            if child["data"]["operationType"] == "Output":
                return True
                
            q.append(child)

    return False

def toAdjList(nodesList):

    cipherID  = {}
    cipherIdx = {}
    for i, node in enumerate(nodesList):
        cipherID[node["id"]] = i
        cipherIdx[i] = node["id"]

    adj = [[] for _ in range(len(nodesList))]
    for node in nodesList:
        for c in node["children"]:
            adj[cipherID[node["id"]]].append(cipherID[c])
    
    print(adj)

    return adj, cipherID, cipherIdx


def isCyclic(adj, idxToIdCipher):

    # find all bridges
    # once you have all bridges, remove them
    # any nodes that still have incoming/outgoing edges
    # are part of a cycle

    """
    NOTES:
        https://en.wikipedia.org/wiki/Bridge_%28graph_theory%29
        https://cs.stackexchange.com/questions/92827/graph-find-all-vertices-that-are-part-of-a-cycle
    """


    pass

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
        errors.append(ParseError("No \'Input Node\' defined", nodeIDs=None))

    # checks for input's Having a parent
    for n in inputs:
        if len(n.get("parents", [])) != 0:
            errors.append(ParseError("Inputs cannot have a parent", nodeIDs=[n["id"]]))

    # checks for maxInputs and minInputs being obeyed
    # print(" ".join([n["id"] for n in nodesList]))
    for n in nodesList:
        if n["data"]["operationType"] == "Input": continue # checked elsewhere
        if n["data"]["operationType"] == "Output": continue # has no default

        # print(n["parents"])

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
            errors.append(ParseError(f"{dflt['type']} requires at least {min_inputs} input{'s' if min_inputs > 1 else ''}, currently has {parent_count}", nodeIDs=[n["id"]]))
        if parent_count > int(max_inputs):
            errors.append(ParseError(f"{dflt['type']} requires less or equal to {max_inputs} input{'s' if max_inputs > 1 else ''}, currently has {parent_count}", nodeIDs=[n["id"]]))

    # get all inputs for graph
    inputs = [n for n in nodesList if n["data"]["operationType"] == "Input"]
   
    pathCheck = [(dfs(nodesList, i), i) for i in inputs]
    for path in pathCheck:
        if not path[0]: errors.append(ParseError(f"This input has no path to an output", nodeIDs=[path[1]["id"]]))

    # TODO: checks for matching number of dimensions from output to input

    # TODO: check output does not have too many or too little inputs.

    # TODO: check for cycles.

    if nodesList.__len__() > 4:
        adj, idToIdxCipher, idxToIdCipher = toAdjList(nodesList)
        if sum(sum(r) for r in adj) != 0:
            isCyclic(adj, idxToIdCipher)

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