import torch
import torch.onnx
from model import Net

# Initialize the model
net = Net()

# Load the model checkpoint
checkpoint = torch.load('model_checkpoint.pth')

# Check if the checkpoint contains 'model_state_dict' and load it
if 'model_state_dict' in checkpoint:
    net.load_state_dict(checkpoint['model_state_dict'])
else:
    net.load_state_dict(checkpoint)  # Assuming the entire checkpoint is the state dictionary

# Set the model to evaluation mode
net.eval()

# Create example input data
dummy_input = torch.randn(1, 3, 32, 32)

# Export the model
torch.onnx.export(net, dummy_input, "model.onnx", export_params=True, opset_version=10, do_constant_folding=True, input_names=['input'], output_names=['output'])
