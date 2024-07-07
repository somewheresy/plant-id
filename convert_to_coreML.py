import torch
import coremltools as ct
from clip import clip
import numpy as np

print(f"torch version: {torch.__version__}")
print(f"coremltools version: {ct.__version__}")
print(f"numpy version: {np.__version__}")

# Load the CLIP model
device = "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Create a new model with the classification head
class PlantClassifier(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.classification_head = torch.nn.Linear(clip_model.visual.output_dim, 1081)  # 1081 classes based on the error message

    def forward(self, x):
        features = self.clip_model.encode_image(x)
        return self.classification_head(features)

model = PlantClassifier(clip_model)

# Load the trained weights
state_dict = torch.load("plant_id_model_final.pth", map_location=device)

# Create a new state dict with the keys we need
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("visual.") or k == "positional_embedding" or k == "logit_scale":
        new_state_dict["clip_model." + k] = v
    elif k == "classification_head.weight" or k == "classification_head.bias":
        new_state_dict[k] = v

# Load the new state dict
model.load_state_dict(new_state_dict, strict=False)

model.eval()

# Trace the model with an example input
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# Convert to CoreML
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="image", shape=example_input.shape)],
    outputs=[ct.TensorType(name="output")],
    minimum_deployment_target=ct.target.iOS15,
    convert_to="mlprogram"
)

# Save the CoreML model
mlmodel.save("PlantClassifier.mlpackage")

print("Conversion complete. Model saved as PlantClassifier.mlpackage")
