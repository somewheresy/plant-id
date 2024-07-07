import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import clip
from clip import clip
from PIL import Image
from tqdm import tqdm
import os
import multiprocessing

# Ensure proper multiprocessing behavior on macOS
multiprocessing.set_start_method('spawn', force=True)

def main():
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load the dataset
    dataset = load_dataset("mikehemberger/plantnet300k")

    # Custom collate function
    def collate_fn(batch):
        images = [preprocess(item['image'].convert("RGB")) for item in batch]
        labels = [item['label'] for item in batch]
        return {
            'image': torch.stack(images),
            'label': torch.tensor(labels)
        }

    # Create DataLoader
    train_loader = DataLoader(
        dataset['train'], 
        batch_size=32, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )

    # Load the previously trained model
    num_classes = len(dataset['train'].features['label'].names)
    model.classification_head = nn.Linear(model.visual.output_dim, num_classes).to(device)
    model.load_state_dict(torch.load("plant_id_model_final.pth", map_location=device))

    # Unfreeze CLIP parameters for fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)  # Lower learning rate for fine-tuning
    criterion = nn.CrossEntropyLoss()

    # Continued training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            image_features = model.encode_image(images)
            outputs = model.classification_head(image_features)
            
            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Print epoch stats
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save model
        torch.save(model.state_dict(), f"plant_id_model_finetuned_epoch_{epoch+1}.pth")

    print("Fine-tuning complete!")

    # Save final fine-tuned model
    torch.save(model.state_dict(), "plant_id_model_finetuned_final.pth")

if __name__ == '__main__':
    main()