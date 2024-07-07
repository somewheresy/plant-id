import torch

state_dict = torch.load("plant_id_model_final.pth", map_location="cpu")
print(state_dict.keys())