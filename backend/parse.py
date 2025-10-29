
from NNdatabase import NNDataBase

CRUDE_REPORT = True
DEBUG = True

class ParseError:

    def __init__(self, desc, nodes:list=None):
        self.desc = desc
        self.nodes     = None if nodes is None else [n["id"] for n in nodes]
        self.nodesDesc = None if nodes is None else [f"{n["data"]["label"]} {n["id"]}" for n in nodes]

    def report(self, ID):
        nodesS = "No nodes"
        if self.nodesDesc is not None:
            nodesS = ", ".join(self.nodes)
        return f"{ID}: {self.desc}, involving \'{nodesS}\'"

def find(nodesList, nodeId):
    query = [n for n in nodesList if n["id"] == nodeId]
    return None if len(query) == 0 else query[0]

def dfs(nodesList, origin):

    q = [origin]
    v = [] # id list

    while len(q) != 0:

        n = q.pop()

        assert isinstance(n, dict)

        if n["id"] in v: continue
        v.append(n["id"])

        # assumes children list contains only ID's

        for c in n["children"]:
            child = find(nodesList, c)
            if child is None: 
                print(f"couldn't find {c} in nodes list")
                continue
            assert isinstance(child, dict)
            if child["data"]["operationType"] == "Output":
                if CRUDE_REPORT: print(f"Input {origin["id"]} found an output")
                return True
            q.append(child)

    return False

def parse(nodesList):
    
    # errors list
    """
        every error needs to have a unique ID and description, as errors are fixed, the ID's
        need to change, storing in order in a queue is probably the best way.
    """
    DB = NNDataBase() # this object is shit, fixed below
    defaults = DB.defaults["nnLayer"]["data"] + DB.defaults["tensorOp"]["data"] + DB.defaults["activator"]["data"]
    errors = []

    inputs = [n for n in nodesList if (n["data"]["operationType"]).lower() == "input"]
    # checks for input existing
    if len(inputs) == 0:
        errors.append(ParseError("No Input Node, defined", nodes=None))

    # checks for input's Having a parent
    for n in inputs:
        if len(n["parents"]) != 0:
            errors.append(ParseError("Inputs cannot have a Parent", nodes=[n]))

    # checks for maxInputs and minInputs being obeyed
    print(" ".join([n["id"] for n in nodesList]))
    for n in nodesList:
        if n["data"]["operationType"] == "Input": continue # checked elsewhere
        if n["data"]["operationType"] == "Output": continue # has no default

        # print(n["parents"])

        dflt = [d for d in defaults if d["type"] == n["data"]["label"]][0]
        if len(n["parents"]) < int(dflt["minInputs"]):
            errors.append(ParseError(f"Node of Type {dflt["type"]} requires at least {dflt["minInputs"]} input{"s" if dflt["minInputs"] > 1 else ""}, currently has {len(n["parents"])}", nodes=[n]))
        if len(n["parents"]) > int(dflt["maxInputs"]):
            errors.append(ParseError(f"Node of type {dflt["type"]} requires less or equal to {dflt["maxInputs"]} input{"s" if dflt["minInputs"] > 1 else ""}, currently has {len(n["parents"])}", nodes=[n]))

    # TODO: checks for path from input to output
    inputs = [n for n in nodesList if n["data"]["operationType"] == "Input"]
    # print(inputs)
    pathCheck = [(dfs(nodesList, i), i) for i in inputs]
    for path in pathCheck:
        if not path[0]: errors.append(ParseError(f"Input \'{path[1]["id"]}\' has no path to output", nodes=[path[1]]))

    # TODO: checks for matching number of dimensions from output to input

    if CRUDE_REPORT:
        for i, e in enumerate(errors):
            print(e.report(i + 1))

    return [] if len(errors) == 0 else [
        {
            "errorMsg" : e.report(i + 1),
            "flaggedNodes" : e.nodes
        }
        for i, e in enumerate(errors)
    ]