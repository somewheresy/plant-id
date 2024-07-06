import torch
import coremltools as ct

# Load your trained model
model = torch.load("plant_id_model_final.pth", map_location=torch.device('cpu'))
model.eval()

# Trace the model with an example input
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# Convert to CoreML
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="image", shape=example_input.shape)],
    outputs=[ct.TensorType(name="output")]
)

# Save the CoreML model
mlmodel.save("PlantClassifier.mlmodel")