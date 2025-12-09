"""
Script d'entraînement principal
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
import os
import sys

# Ajout du chemin des modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.dataset import FaceForensicsDataset
from models.xception_model import XceptionDeepfakeDetector
from models.model_utils import train_epoch, validate

def parse_args():
    parser = argparse.ArgumentParser(description='Entraînement Xception pour détection deepfake')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Chemin vers le dossier du dataset')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Nombre d\'époques')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Taille du batch')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='trained_models',
                       help='Dossier de sauvegarde des modèles')
    return parser.parse_args()

def prepare_dataloaders(data_path, batch_size=32):
    """Prépare les DataLoaders pour train/val/test"""
    
    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset complet
    full_dataset = FaceForensicsDataset(data_path, train_transform, 'full')
    
    # Split manuel 70-15-15
    total = len(full_dataset)
    indices = torch.randperm(total).tolist()
    
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    # Création des subsets
    train_set = Subset(full_dataset, train_idx)
    val_set = Subset(full_dataset, val_idx)
    test_set = Subset(full_dataset, test_idx)
    
    # Modification des transformations pour val/test
    val_set.dataset.transform = val_transform
    test_set.dataset.transform = val_transform
    
    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size,
                            shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                           shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"\n Split des données:")
    print(f"  Train: {len(train_set)} images")
    print(f"  Validation: {len(val_set)} images")
    print(f"  Test: {len(test_set)} images")
    
    return train_loader, val_loader, test_loader

def main():
    args = parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Device: {device}")
    
    # Dataloaders
    print("\n Chargement des données...")
    train_loader, val_loader, test_loader = prepare_dataloaders(
        args.data_path, args.batch_size
    )
    
    # Modèle
    print("\n Création du modèle...")
    model = XceptionDeepfakeDetector(num_classes=2, pretrained=True)
    model.print_architecture()
    model = model.to(device)
    
    # Loss et optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-5
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    # Entraînement
    print(f"\n Début de l'entraînement ({args.epochs} époques)")
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_acc = validate(model, val_loader, device)
        
        # Scheduler
        scheduler.step(val_acc)
        
        # Historique
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        
        # Affichage
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Acc: {val_acc:.2f}%")
        
        # Sauvegarde du meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.output_dir, f'best_model_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_loss': train_loss,
            }, save_path)
            print(f"   Meilleur modèle sauvegardé: {save_path}")
    
    # Sauvegarde finale
    final_path = os.path.join(args.output_dir, 'xception_model_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\n Entraînement terminé!")
    print(f" Modèle final sauvegardé: {final_path}")
    print(f" Meilleure accuracy validation: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main()