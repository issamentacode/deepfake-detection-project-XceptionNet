"""
Script d'évaluation du modèle
"""

import argparse
import torch
import numpy as np
from sklearn.metrics import (confusion_matrix, classification_report,
                           accuracy_score, f1_score, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.dataset import FaceForensicsDataset
from models.xception_model import XceptionDeepfakeDetector
from torchvision import transforms

def evaluate_model(model, test_loader, device):
    """Évaluation complète du modèle"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probabilité classe 1
    
    # Métriques
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    auc = roc_auc_score(all_labels, all_probs)
    
    # Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    
    # Rapport détaillé
    report = classification_report(all_labels, all_preds,
                                 target_names=['Original', 'Deepfake'])
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'auc_roc': auc,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """Visualise la matrice de confusion"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Original', 'Deepfake'],
                yticklabels=['Original', 'Deepfake'])
    plt.title('Matrice de Confusion')
    plt.ylabel('Vérité Terrain')
    plt.xlabel('Prédiction')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Évaluation du modèle')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Chemin vers le modèle .pth')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Chemin vers les données de test')
    parser.add_argument('--output', type=str, default='results/metrics.txt',
                       help='Fichier de sortie pour les métriques')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Chargement du modèle
    print(f" Chargement du modèle: {args.model_path}")
    model = XceptionDeepfakeDetector(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Transformations pour test
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset de test
    test_dataset = FaceForensicsDataset(args.test_data, transform, 'test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=2
    )
    
    print(f" Évaluation sur {len(test_dataset)} images...")
    
    # Évaluation
    results = evaluate_model(model, test_loader, device)
    
    # Affichage des résultats
    print("\n" + "="*60)
    print(" RÉSULTATS D'ÉVALUATION")
    print("="*60)
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"AUC-ROC: {results['auc_roc']:.4f}")
    print("\n" + results['classification_report'])
    
    # Visualisation
    plot_confusion_matrix(results['confusion_matrix'])
    
    # Sauvegarde des métriques
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write("RÉSULTATS D'ÉVALUATION\n")
        f.write("="*50 + "\n")
        f.write(f"Accuracy: {results['accuracy']*100:.2f}%\n")
        f.write(f"F1-Score: {results['f1_score']:.4f}\n")
        f.write(f"AUC-ROC: {results['auc_roc']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(results['classification_report'])
    
    print(f"\n Métriques sauvegardées: {args.output}")

if __name__ == '__main__':
    main()