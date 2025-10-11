import json

def generate(json):
    header = """class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()\n"""
        
    layers = generateBody(json)
    
    indented_layers = "     " + "\n             ".join(layers.split("\n"))
    init = header + "        self.layers = nn.Sequential(\n        "+ indented_layers + "\n        )"
    
    forward = """\n\n        def forward(self, x):
            out = self.layers(x)
            return out"""
    return init + forward

def generateBody(json):
    
    layers = []
    for layer in json["layers"]: 
        layer_type = layer.get("type")
        layer_param = layer.get("parameters")

        if layer_param:
            args = ", ".join(f"{k} = {v!r}" for k, v in layer_param.items())
            line = f"nn.{layer_type}({args})"
        else:
            line = f"nn.{layer_type}()"
        layers.append(line)
    return ",\n".join(layers)

f = open("JsonLayers/test.json", "r")   # path must be a string
data = json.load(f)
f.close()
print(generate(data))
