import torch.onnx
from model import Net

net = Net()
net.load_state_dict(torch.load('model_checkpoint.pth'))
net.eval()

# Create example input data
dummy_input = torch.randn(1, 3, 32, 32)

# Export the model
torch.onnx.export(net, dummy_input, "model.onnx", export_params=True, opset_version=10, do_constant_folding=True, input_names = ['input'], output_names = ['output'])