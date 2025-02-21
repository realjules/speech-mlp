import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import wandb
from tqdm import tqdm

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for frames, phonemes in progress_bar:
        frames, phonemes = frames.to(device), phonemes.to(device)
        
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, phonemes)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += phonemes.size(0)
        correct += predicted.eq(phonemes).sum().item()
        
        progress_bar.set_postfix({
            'loss': total_loss / (progress_bar.n + 1),
            'acc': 100. * correct / total
        })
    
    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        for frames, phonemes in progress_bar:
            frames, phonemes = frames.to(device), phonemes.to(device)
            
            outputs = model(frames)
            loss = criterion(outputs, phonemes)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += phonemes.size(0)
            correct += predicted.eq(phonemes).sum().item()
            
            progress_bar.set_postfix({
                'loss': total_loss / (progress_bar.n + 1),
                'acc': 100. * correct / total
            })
    
    return total_loss / len(val_loader), 100. * correct / total

def predict(model, test_loader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Testing')
        for frames in progress_bar:
            frames = frames.to(device)
            outputs = model(frames)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
    
    return predictions

def create_optimizer(model, lr=1e-3):
    return AdamW(model.parameters(), lr=lr)

def create_scheduler(optimizer, mode='cosine', **kwargs):
    if mode == 'cosine':
        return CosineAnnealingLR(optimizer, **kwargs)
    elif mode == 'plateau':
        return ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler mode: {mode}")

def setup_wandb(config):
    wandb.init(
        project="speech-recognition-mlp",
        config=config
    )

def log_metrics(metrics, step=None):
    wandb.log(metrics, step=step)