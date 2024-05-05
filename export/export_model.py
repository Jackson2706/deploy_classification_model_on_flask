import torch
from model import  SVHN_Net
model_pth_path = "../weights/trained_model.pth"

model = SVHN_Net()
model.load_state_dict(torch.load(model_pth_path, map_location=torch.device('cpu')))
dummy_input = torch.randn(1, 3, 32, 32)  # Change the shape according to your model's input
torch.onnx.export(model, dummy_input, "weights/cls_model.onnx", verbose=True)
