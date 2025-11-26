# TorchWeaver Test Files

This directory contains test models demonstrating TorchWeaver's code generation capabilities.

## Structure

- `model_pics/` - Visual network diagrams (PNG files) showing the neural network architecture designed in TorchWeaver
- `testing_models/` - Generated Python code files corresponding to each model diagram

## Test Models

Each test model consists of:
1. **PNG diagram** in `model_pics/` - The visual representation from TorchWeaver's canvas
2. **Python code** in `testing_models/` - The generated PyTorch model code

### Model Correspondence

| Diagram | Generated Code | Description |
|---------|---------------|-------------|
| `model_pics/model(160).png` | `testing_models/model(160).py` | Simple single linear layer model |
| `model_pics/model(161).png` | `testing_models/model(161).py` | Multi-path network with concatenation |
| `model_pics/model(162).png` | `testing_models/model(162).py` | Split and merge architecture |

## Testing

All models in `testing_models/` have been **compiled and successfully executed** using PyTorch.

### Running the Tests

To run any test model:

```bash
cd testing_models
python "model(160).py"
python "model(161).py"
python "model(162).py"
```

**Note:** Use quotes around filenames containing parentheses in PowerShell/CMD.

### Expected Results

Each model should:
- ✅ Compile without errors
- ✅ Execute forward pass successfully
- ✅ Print output tensor shape or values
- ✅ Complete training step (if included)

## Verification

All models have been tested and verified to:
1. Generate valid PyTorch code
2. Compile without syntax errors
3. Execute without runtime errors
4. Produce expected tensor shapes
