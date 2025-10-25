import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import make_dataloaders
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import modules
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torchvision.transforms as T

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def plot_training_history(history, save_dir):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 Score
    axes[1, 0].plot(history['train_f1'], label='Train')
    axes[1, 0].plot(history['val_f1'], label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Training and Validation F1')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # AUC
    axes[1, 1].plot(history['train_auc'], label='Train')
    axes[1, 1].plot(history['val_auc'], label='Validation')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC-ROC')
    axes[1, 1].set_title('Training and Validation AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path, class_names=['AD', 'NC']):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_metrics(y_true, y_pred, y_probs):
    """Calculate comprehensive metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    # AUC-ROC (using probability of positive class)
    try:
        auc = roc_auc_score(y_true, y_probs[:, 1])
    except:
        auc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    
    losses = AverageMeter()
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record metrics
        losses.update(loss.item(), images.size(0))
        
        # Get predictions
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
    
    # Calculate metrics
    metrics = calculate_metrics(
        np.array(all_labels), 
        np.array(all_preds),
        np.array(all_probs)
    )
    metrics['loss'] = losses.avg
    
    return metrics


def validate(model, loader, criterion, device, epoch):
    """Validate the model."""
    model.eval()
    
    losses = AverageMeter()
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Val]')
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Record metrics
            losses.update(loss.item(), images.size(0))
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
    
    # Calculate metrics
    metrics = calculate_metrics(
        np.array(all_labels), 
        np.array(all_preds),
        np.array(all_probs)
    )
    metrics['loss'] = losses.avg
    
    return metrics, np.array(all_labels), np.array(all_preds)

def train(data_dir='ADNI/AD_NC', batch_size=32, num_epochs=10, lr=3e-5, device='cuda', save_dir='checkpoints', resume_from=None, drop_path_rate=0.1, layer_scale=1e-6, weight_decay=0.1):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Dataloaders
    train_loader, test_loader = make_dataloaders(data_dir, batch_size=batch_size, img_size=224)
    if resume_from is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(save_dir, f'ConvNeXt_a_bit_smaller_{timestamp}')
        os.makedirs(save_dir, exist_ok=True)
    else:
        # If resuming, use the same directory as the checkpoint
        save_dir = os.path.dirname(resume_from)
    # Model
    model = modules.ConvNeXt(
        in_chans=1,
        depths=[2, 2, 6, 2],
        dims=[64, 128, 256, 512],
        num_classes=2,
        drop_path_prob=drop_path_rate,
        layer_scale_init=layer_scale
    ).to(device)

    best_val_acc = 0.0
    start_epoch = 1

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay,betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    if resume_from is not None:
        start_epoch, best_val_acc, loaded_metrics = load_checkpoint(
            resume_from, model, optimizer, scheduler, device
        )
        start_epoch += 1  # Start from next epoch
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
        'train_auc': [], 'val_auc': []
    }
    for epoch in range(start_epoch, num_epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_metrics, val_labels, val_preds = validate(model, test_loader, criterion, device, epoch)
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_auc'].append(train_metrics['auc'])
        history['val_auc'].append(val_metrics['auc'])
        
        # Print metrics
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        is_best = val_metrics['accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['accuracy']
        
        save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, save_dir, is_best)
    
    plot_training_history(history, save_dir)
    plot_confusion_matrix(val_labels, val_preds, os.path.join(save_dir, 'confusion_matrix_final.png'))
    print(f"\nResults saved to: {save_dir}")
    
    return model, history

def validate(model, loader, criterion, device, epoch):
    """Validate the model."""
    model.eval()
    
    losses = AverageMeter()
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Val]')
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Record metrics
            losses.update(loss.item(), images.size(0))
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
    
    # Calculate metrics
    metrics = calculate_metrics(
        np.array(all_labels), 
        np.array(all_preds),
        np.array(all_probs)
    )
    metrics['loss'] = losses.avg
    
    return metrics, np.array(all_labels), np.array(all_preds)

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, os.path.join(save_dir, 'latest.pth'))
    
    if is_best:
        torch.save(checkpoint, os.path.join(save_dir, 'best.pth'))
        print(f"Saved best model (Accuracy: {metrics['accuracy']:.4f})")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cuda'):
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Get epoch and metrics
    start_epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    best_val_acc = metrics.get('accuracy', 0.0)
    
    print(f"Resuming from epoch {start_epoch}")
    print(f"Previous best accuracy: {best_val_acc:.4f}")
    
    return start_epoch, best_val_acc, metrics

if __name__ == "__main__":
    # Train model
    if len(sys.argv) > 1:
        if sys.argv[1] == '--checkpoint' and len(sys.argv) > 2:
            checkpoint_path = sys.argv[2]
            model, history = train(data_dir='ADNI/AD_NC', batch_size=16, num_epochs=100, lr=1e-4, drop_path_rate=0.3, layer_scale=1e-6, weight_decay=0.2, resume_from=checkpoint_path)
        else:
            print("Usage: python train.py [--checkpoint checkpoint_path]")
    else:
        model, history = train(data_dir='ADNI/AD_NC', batch_size=16, num_epochs=100, lr=1e-4, drop_path_rate=0.3, layer_scale=1e-6, weight_decay=0.2)