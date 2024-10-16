import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import ViTFeatureExtractor, RobertaTokenizer, ViTModel, RobertaModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x, context):
        q = self.query(x)
        k = self.key(context)
        v = self.value(context)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return out

class CrossViTBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, context):
        x = x + self.attn(self.norm1(x), self.norm1(context))
        x = x + self.ffn(self.norm2(x))
        return x

class CrossViT(nn.Module):
    def __init__(self, img_dim, text_dim, hidden_dim, num_layers):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        self.img_blocks = nn.ModuleList([CrossViTBlock(hidden_dim) for _ in range(num_layers)])
        self.text_blocks = nn.ModuleList([CrossViTBlock(hidden_dim) for _ in range(num_layers)])

    def forward(self, img_features, text_features):
        img = self.img_proj(img_features)
        text = self.text_proj(text_features)

        for img_block, text_block in zip(self.img_blocks, self.text_blocks):
            img = img_block(img, text)
            text = text_block(text, img)

        return img, text
class ADEClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        
        self.cross_vit = CrossViT(768, 768, 512, num_layers=3)
        
        self.shared_classifier = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        self.img_specific_classifier = nn.Linear(768, num_classes)
        self.text_specific_classifier = nn.Linear(768, num_classes)
        
        self.missing_modality_generator = nn.Linear(512, 512)

    def forward(self, image, input_ids, attention_mask, fine_tuned_input_ids, fine_tuned_attention_mask, image_mask, text_mask):
        batch_size = image.size(0)
        
        # Process image
        img_features = self.vit(pixel_values=image).last_hidden_state
        img_features = img_features * image_mask.unsqueeze(1).unsqueeze(2)
        
        # Process original text
        text_features = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_features = text_features * text_mask.unsqueeze(1).unsqueeze(2)
        
        # Process fine-tuned text
        fine_tuned_features = self.roberta(input_ids=fine_tuned_input_ids, attention_mask=fine_tuned_attention_mask).last_hidden_state
        
        # Use fine-tuned text when image is available
        text_features = torch.where(image_mask.unsqueeze(1).unsqueeze(2),
                                    fine_tuned_features,
                                    text_features)

        fused_img, fused_text = self.cross_vit(img_features, text_features)
        
        img_cls = fused_img[:, 0]
        text_cls = fused_text[:, 0]
        
        # Generate missing modality features
        missing_img_features = self.missing_modality_generator(text_cls) * (~image_mask).unsqueeze(1)
        missing_text_features = self.missing_modality_generator(img_cls) * (~text_mask).unsqueeze(1)
        
        # Combine with original features
        img_cls = img_cls + missing_img_features
        text_cls = text_cls + missing_text_features
        
        combined = torch.cat((img_cls, text_cls), dim=1)
        shared_output = self.shared_classifier(combined)
        
        img_specific_output = self.img_specific_classifier(img_features[:, 0])
        text_specific_output = self.text_specific_classifier(text_features[:, 0])
        
        return shared_output, img_specific_output, text_specific_output, img_cls, text_cls

class ADEDataset(Dataset):
    def __init__(self, csv_file, img_dir, mask_prob=0.3):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.image_processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        text = self.data.iloc[idx, 1]
        blip_fine_tuned_final = self.data.iloc[idx, -1]
        label = 1 if self.data.iloc[idx, 2] == 'ADR' else 0

        # Process image
        image = self.image_processor(images=image, return_tensors="pt")['pixel_values'].squeeze()

        # Process text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # Process blip fine-tuned final text
        blip_fine_tuned_encoding = self.tokenizer.encode_plus(
            blip_fine_tuned_final,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        blip_fine_tuned_input_ids = blip_fine_tuned_encoding['input_ids'].squeeze()
        blip_fine_tuned_attention_mask = blip_fine_tuned_encoding['attention_mask'].squeeze()

        # Randomly mask one modality
        mask_image = random.random() < self.mask_prob
        mask_text = random.random() < self.mask_prob
        
        if mask_image and mask_text:
            mask_text = False  # Ensure at least one modality is always present

        return {
            'image': image if not mask_image else torch.zeros_like(image),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'fine_tuned_input_ids': blip_fine_tuned_input_ids,
            'fine_tuned_attention_mask': blip_fine_tuned_attention_mask,
            'image_mask': torch.tensor(not mask_image, dtype=torch.bool),
            'text_mask': torch.tensor(not mask_text, dtype=torch.bool),
            'label': torch.tensor(label, dtype=torch.long)
        }

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x, context):
        q = self.query(x)
        k = self.key(context)
        v = self.value(context)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return out

class CrossViTBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, context):
        x = x + self.attn(self.norm1(x), self.norm1(context))
        x = x + self.ffn(self.norm2(x))
        return x

class CrossViT(nn.Module):
    def __init__(self, img_dim, text_dim, hidden_dim, num_layers):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        self.img_blocks = nn.ModuleList([CrossViTBlock(hidden_dim) for _ in range(num_layers)])
        self.text_blocks = nn.ModuleList([CrossViTBlock(hidden_dim) for _ in range(num_layers)])

    def forward(self, img_features, text_features):
        img = self.img_proj(img_features)
        text = self.text_proj(text_features)

        for img_block, text_block in zip(self.img_blocks, self.text_blocks):
            img = img_block(img, text)
            text = text_block(text, img)

        return img, text

class ADEClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        
        self.cross_vit = CrossViT(768, 768, 512, num_layers=3)
        
        self.shared_classifier = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        self.img_specific_classifier = nn.Linear(768, num_classes)
        self.text_specific_classifier = nn.Linear(768, num_classes)
        
        self.missing_modality_generator = nn.Linear(512, 512)

    def forward(self, image, input_ids, attention_mask, fine_tuned_input_ids, fine_tuned_attention_mask, image_mask, text_mask):
        batch_size = image.size(0)
        
        # Process image
        img_features = self.vit(pixel_values=image).last_hidden_state
        img_features = img_features * image_mask.unsqueeze(1).unsqueeze(2)
        
        # Process original text
        text_features = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_features = text_features * text_mask.unsqueeze(1).unsqueeze(2)
        
        # Process fine-tuned text
        fine_tuned_features = self.roberta(input_ids=fine_tuned_input_ids, attention_mask=fine_tuned_attention_mask).last_hidden_state
        
        # Use fine-tuned text when image is available
        text_features = torch.where(image_mask.unsqueeze(1).unsqueeze(2),
                                    fine_tuned_features,
                                    text_features)

        fused_img, fused_text = self.cross_vit(img_features, text_features)
        
        img_cls = fused_img[:, 0]
        text_cls = fused_text[:, 0]
        
        # Generate missing modality features
        missing_img_features = self.missing_modality_generator(text_cls) * (~image_mask).unsqueeze(1)
        missing_text_features = self.missing_modality_generator(img_cls) * (~text_mask).unsqueeze(1)
        
        # Combine with original features
        img_cls = img_cls + missing_img_features
        text_cls = text_cls + missing_text_features
        
        combined = torch.cat((img_cls, text_cls), dim=1)
        shared_output = self.shared_classifier(combined)
        
        img_specific_output = self.img_specific_classifier(img_features[:, 0])
        text_specific_output = self.text_specific_classifier(text_features[:, 0])
        
        return shared_output, img_specific_output, text_specific_output, img_cls, text_cls

class ADEDataset(Dataset):
    def __init__(self, csv_file, img_dir, mask_prob=0.3):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.image_processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        text = self.data.iloc[idx, 1]
        blip_fine_tuned_final = self.data.iloc[idx, -1]
        label = 1 if self.data.iloc[idx, 2] == 'ADR' else 0

        # Process image
        image = self.image_processor(images=image, return_tensors="pt")['pixel_values'].squeeze()

        # Process text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # Process blip fine-tuned final text
        blip_fine_tuned_encoding = self.tokenizer.encode_plus(
            blip_fine_tuned_final,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        blip_fine_tuned_input_ids = blip_fine_tuned_encoding['input_ids'].squeeze()
        blip_fine_tuned_attention_mask = blip_fine_tuned_encoding['attention_mask'].squeeze()

        # Randomly mask one modality
        mask_image = random.random() < self.mask_prob
        mask_text = random.random() < self.mask_prob
        
        if mask_image and mask_text:
            mask_text = False  # Ensure at least one modality is always present

        return {
            'image': image if not mask_image else torch.zeros_like(image),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'fine_tuned_input_ids': blip_fine_tuned_input_ids,
            'fine_tuned_attention_mask': blip_fine_tuned_attention_mask,
            'image_mask': torch.tensor(not mask_image, dtype=torch.bool),
            'text_mask': torch.tensor(not mask_text, dtype=torch.bool),
            'label': torch.tensor(label, dtype=torch.long)
        }
def collate_fn(batch):
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'fine_tuned_input_ids': torch.stack([item['fine_tuned_input_ids'] for item in batch]),
        'fine_tuned_attention_mask': torch.stack([item['fine_tuned_attention_mask'] for item in batch]),
        'image_mask': torch.stack([item['image_mask'] for item in batch]),
        'text_mask': torch.stack([item['text_mask'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch])
    }

def train_epoch(model, dataloader, criterion, optimizer, device, lambda_align=0.1):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        shared_output, img_specific_output, text_specific_output, img_cls, text_cls = model(
            image=batch['image'],
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            fine_tuned_input_ids=batch['fine_tuned_input_ids'],
            fine_tuned_attention_mask=batch['fine_tuned_attention_mask'],
            image_mask=batch['image_mask'],
            text_mask=batch['text_mask']
        )
        
        shared_loss = criterion(shared_output, batch['label'])
        img_specific_loss = criterion(img_specific_output, batch['label'])
        text_specific_loss = criterion(text_specific_output, batch['label'])
        
        # Distribution alignment loss
        align_loss = F.mse_loss(img_cls, text_cls)
        
        loss = shared_loss + 0.2 * img_specific_loss + 0.2 * text_specific_loss + lambda_align * align_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            shared_output, _, _, _, _ = model(
                image=batch['image'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                fine_tuned_input_ids=batch['fine_tuned_input_ids'],
                fine_tuned_attention_mask=batch['fine_tuned_attention_mask'],
                image_mask=batch['image_mask'],
                text_mask=batch['text_mask']
            )
            
            loss = criterion(shared_output, batch['label'])
            total_loss += loss.item()
            
            preds = torch.argmax(shared_output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    
    return total_loss / len(dataloader), accuracy, f1, recall, precision

def plot_training_history(train_losses, val_losses, val_accuracies, val_f1_scores):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_accuracies, 'g-')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_f1_scores, 'm-')
    plt.title('Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')

    plt.tight_layout()
    plt.savefig('missing_modality_proposed_final_training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('missing_modality_proposed_final_confusion_matrix.png')
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = ADEDataset(csv_file='blip_fine_tuned_with_captions.csv', img_dir='combined_adr_images', mask_prob=0.3)
    
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=32, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=32, collate_fn=collate_fn)

    model = ADEClassifier(num_classes=2).to(device)

    labels = [data['label'].item() for data in train_data]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    num_epochs = 50
    best_val_accuracy = 0
    train_losses, val_losses, val_accuracies, val_f1_scores = [], [], [], []

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, val_f1, val_recall, val_precision = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
        print(f"Val Recall: {val_recall:.4f}, Val Precision: {val_precision:.4f}")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
        
        print()

    # Plot training history
    plot_training_history(train_losses, val_losses, val_accuracies, val_f1_scores)

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_accuracy, test_f1, test_recall, test_precision = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
    print(f"Test Recall: {test_recall:.4f}, Test Precision: {test_precision:.4f}")

    # Generate predictions for confusion matrix
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            shared_output, _, _, _, _ = model(
                image=batch['image'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                fine_tuned_input_ids=batch['fine_tuned_input_ids'],
                fine_tuned_attention_mask=batch['fine_tuned_attention_mask'],
                image_mask=batch['image_mask'],
                text_mask=batch['text_mask']
            )
            preds = torch.argmax(shared_output, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds)

    print("Training completed. Plots have been saved.")

if __name__ == "__main__":
    main()
