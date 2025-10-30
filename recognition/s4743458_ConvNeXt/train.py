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
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Validation', linewidth=2)
    axes[0, 1].axhline(y=0.5, color='r', linestyle='--', label='Random', alpha=0.5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(history['train_f1'], label='Train', linewidth=2)
    axes[1, 0].plot(history['val_f1'], label='Validation', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Training and Validation F1')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # AUC
    axes[1, 1].plot(history['train_auc'], label='Train', linewidth=2)
    axes[1, 1].plot(history['val_auc'], label='Validation', linewidth=2)
    axes[1, 1].axhline(y=0.5, color='r', linestyle='--', label='Random', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC-ROC')
    axes[1, 1].set_title('Training and Validation AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path, class_names=['AD', 'NC']):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    # Add percentages
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j+0.5, i+0.7, f'({cm[i,j]/cm[i].sum()*100:.1f}%)',
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
def calculate_metrics(y_true, y_pred, y_probs):
    """Calculate comprehensive metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(y_true, y_probs[:, 1])
    except:
        auc = 0.5
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=25, min_delta=0.005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_metric):
        score = val_metric
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop



def train_epoch(model, loader, criterion, optimizer, device, epoch, scaler=None):
    """Train for one epoch - simplified without mixup initially."""
    model.train()
    
    losses = AverageMeter()
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Record metrics
        losses.update(loss.item(), images.size(0))
        
        # Get predictions
        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        
        # Update progress bar with gradient norm
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracy_score(all_labels, all_preds):.3f}'
        })
    
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
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            losses.update(loss.item(), images.size(0))
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
    
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


def train(data_dir='ADNI/AD_NC', 
          batch_size=16, 
          num_epochs=200, 
          lr=3e-4,
          device='cuda', 
          save_dir='checkpoints', 
          resume_from=None, 
          drop_path_rate=0.2,
          layer_scale=1e-6, 
          weight_decay=0.02,
          patience=30,
          use_amp=True):
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Create save directory
    if resume_from is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(save_dir, f'convnext_friday_{timestamp}')
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = os.path.dirname(resume_from)
    
    # Dataloaders
    train_loader, test_loader = make_dataloaders(
        data_dir, 
        batch_size=batch_size, 
        img_size=224,
    )
    
    # Model with REDUCED regularization
    print("Creating model...")
    model = modules.convnext_small_2(
        drop_path_rate=drop_path_rate, 
        layer_scale=layer_scale,
    ).to(device)

    
    # Simple CrossEntropy (NO label smoothing initially)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer with moderate weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if use_amp and torch.cuda.is_available() else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=0.005)
    
    # Resume from checkpoint
    best_val_acc = 0.0
    start_epoch = 1
    if resume_from is not None:
        start_epoch, best_val_acc, _ = load_checkpoint(
            resume_from, model, optimizer, scheduler, device
        )
        start_epoch += 1
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
        'train_auc': [], 'val_auc': []
    }
    
    print(f"{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Epochs:           {num_epochs}")
    print(f"Batch size:       {batch_size}")
    print(f"Learning rate:    {lr}")
    print(f"Weight decay:     {weight_decay}")
    print(f"Drop path rate:   {drop_path_rate}")
    print(f"Early stopping:   {patience} epochs")
    print(f"Mixed precision:  {use_amp}")
    print(f"Scheduler:        ReduceLROnPlateau")
    print(f"{'='*60}\n")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler
        )
        
        # Validate
        val_metrics, val_labels, val_preds = validate(
            model, test_loader, criterion, device, epoch
        )
        
        # Update scheduler based on validation accuracy
        scheduler.step(val_metrics['accuracy'])
        
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
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")
        print(f"Train - Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | "
              f"F1: {train_metrics['f1']:.4f} | AUC: {train_metrics['auc']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save checkpoint
        is_best = val_metrics['accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['accuracy']
            print(f"\n🎉 New best validation accuracy: {best_val_acc:.4f}")
        
        save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, save_dir, is_best)
        
        # Plot every 10 epochs
        if epoch % 10 == 0:
            plot_training_history(history, save_dir)
            plot_confusion_matrix(val_labels, val_preds, 
                                os.path.join(save_dir, f'confusion_matrix_epoch{epoch}.png'))
        
        # Early stopping check
        if early_stopping(val_metrics['accuracy']):
            print(f"\n{'='*60}")
            print(f"Early stopping triggered after {epoch} epochs")
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            print(f"{'='*60}")
            break
        
        print(f"{'='*60}\n")
    
    # Final plots
    plot_training_history(history, save_dir)
    plot_confusion_matrix(val_labels, val_preds, 
                         os.path.join(save_dir, 'confusion_matrix_final.png'))
    
    print(f"\n{'='*60}")
    print(f"Training Completed!")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*60}\n")
    
    return model, history


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '--checkpoint' and len(sys.argv) > 2:
            checkpoint_path = sys.argv[2]
            model, history = train(
                data_dir='ADNI/AD_NC',
                batch_size=16,
                num_epochs=60,
                lr=1e-4, 
                drop_path_rate=0.2,
                layer_scale=1e-6,
                weight_decay=0.05,
                resume_from=checkpoint_path,
                patience=30,
                use_amp=True
            )
        else:
            print("Usage: python train.py [--checkpoint checkpoint_path]")
    else:
        model, history = train(
            data_dir='ADNI/AD_NC',
            batch_size=16,
            num_epochs=60,
            lr=1e-4,
            drop_path_rate=0.2,
            layer_scale=1e-6,
            weight_decay=0.05,
            patience=30,
            use_amp=True
        )