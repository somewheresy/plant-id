import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import clip
from clip import clip
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(model_path, batch_size=32):
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load the dataset
    dataset = load_dataset("mikehemberger/plantnet300k")
    
    # Add classification head
    num_classes = len(dataset['train'].features['label'].names)
    model.classification_head = nn.Linear(model.visual.output_dim, num_classes).to(device)

    # Load the fine-tuned model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Custom collate function
    def collate_fn(batch):
        images = [preprocess(item['image'].convert("RGB")) for item in batch]
        labels = [item['label'] for item in batch]
        return {
            'image': torch.stack(images),
            'label': torch.tensor(labels)
        }

    # Create DataLoader for validation set
    val_loader = DataLoader(
        dataset['validation'], 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            image_features = model.encode_image(images)
            outputs = model.classification_head(image_features)
            
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert class indices to class names
    class_names = dataset['train'].features['label'].names
    y_true = [class_names[label] for label in all_labels]
    y_pred = [class_names[pred] for pred in all_predictions]

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Generate classification report
    report = classification_report(y_true, y_pred)
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    model_path = "plant_id_model_final.pth"  # or use a specific epoch model
    evaluate_model(model_path)