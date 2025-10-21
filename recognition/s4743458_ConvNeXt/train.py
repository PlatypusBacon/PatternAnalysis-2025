import torch
import torch.nn as nn
import torch.optim as optim
from dataset import make_dataloaders
import modules

def train(data_dir='ADNI/AD_NC', batch_size=16, num_epochs=10, lr=1e-4, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Dataloaders
    train_loader, val_loader = make_dataloaders(data_dir, batch_size=batch_size, img_size=224)
    
    # Model
    model = modules.ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        num_classes=2
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_loss /= total
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_loss /= total
        val_acc = correct / total
        
        print(f"Epoch {epoch}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    return model

if __name__ == "__main__":
    model = train()
