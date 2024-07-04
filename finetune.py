import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import clip
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("mikehemberger/plantnet300K")

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Freeze CLIP parameters
for param in model.parameters():
    param.requires_grad = False

# Add a new classification head
num_classes = 1081  # Number of plant species in PlantNet300K
model.classification_head = nn.Linear(model.visual.output_dim, num_classes).to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.classification_head.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch in DataLoader(dataset['train'], batch_size=32, shuffle=True):
        images = torch.stack([transform(Image.open(img_path).convert("RGB")) for img_path in batch['image_path']]).to(device)
        labels = torch.tensor(batch['label']).to(device)
        
        # Forward pass
        with torch.no_grad():
            image_features = model.encode_image(images)
        outputs = model.classification_head(image_features)
        
        # Backward pass and optimize
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Save the fine-tuned model
torch.save(model.state_dict(), "plant_id_model.pth")