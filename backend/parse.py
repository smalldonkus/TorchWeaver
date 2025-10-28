
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
        return f"{ID}: {self.desc}, involving {nodesS}"

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
    for n in nodesList:
        if n["data"]["operationType"] == "Input": continue # checked elsewhere
        dflt = [d for d in defaults if d["type"] == n["data"]["label"]][0]
        if len(n["parents"]) < int(dflt["minInputs"]):
            errors.append(ParseError(f"Node of Type {dflt["type"]} requires at least {dflt["minInputs"]} inputs", nodes=[n]))
        if len(n["parents"]) > int(dflt["maxInputs"]):
            errors.append(ParseError(f"Node of Type {dflt["type"]} requires less than {dflt["maxInputs"]} inputs", nodes=[n]))

    # TODO: checks for path from input ot output

    # TODO: checks for matching number of dimensions from output to input

    if CRUDE_REPORT:
        for ID in range(len(errors)):
            print(errors[ID].report(ID + 1))

    return [] if len(errors) == 0 else [
        {
            "errorMsg" : e.report(i + 1),
            "flaggedNodes" : e.nodes
        }
        for i, e in enumerate(errors)
    ]