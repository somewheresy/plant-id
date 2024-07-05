import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import clip
from clip import clip
from datasets import load_dataset
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the dataset
dataset = load_dataset("mikehemberger/plantnet300K", "default")
# Apply the transform to the dataset
model, preprocess = clip.load("ViT-B/32", device=device)
# Define a custom transform
def transform_image(example):
    if isinstance(example['image'], list):
        # Handle batch of images
        return {
            'image': [preprocess(img.convert("RGB")) for img in example['image']],
            'label': example['label']
        }
    else:
        # Handle single image
        return {
            'image': preprocess(example['image'].convert("RGB")),
            'label': example['label']
        }

dataset = dataset.map(transform_image, batched=True, batch_size=100)

# Set the format of the dataset to PyTorch tensors
dataset.set_format(type='torch', columns=['image', 'label'])

# Load CLIP model
# Freeze CLIP parameters
for param in model.parameters():
    param.requires_grad = False

# Add a new classification head
num_classes = 1081  # Number of plant species in PlantNet300K
model.classification_head = nn.Linear(model.visual.output_dim, num_classes).to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.classification_head.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Create DataLoader
train_loader = DataLoader(dataset['train'], batch_size=32, shuffle=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for i, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        image_features = model.encode_image(images)
        outputs = model.classification_head(image_features)
        
        # Backward pass and optimize
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print some info
        print(f"Processed batch {i+1}. Image features shape: {image_features.shape}")
        
        # Break after a few batches for testing
        if i == 5:
            break
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Save the fine-tuned model
torch.save(model.state_dict(), "plant_id_model.pth")