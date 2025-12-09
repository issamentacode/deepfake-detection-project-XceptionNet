"""
Xception Model for Deepfake Detection
Fine-tuning d'un Xception pré-entraîné sur ImageNet
"""

import torch
import torch.nn as nn
import timm

class XceptionDeepfakeDetector(nn.Module):
    """Classificateur de deepfakes basé sur Xception"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(XceptionDeepfakeDetector, self).__init__()
        
        # Chargement du backbone Xception
        self.backbone = timm.create_model('xception', pretrained=pretrained)
        
        # Remplacement de la tête de classification
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Stratégie de fine-tuning
        self._setup_fine_tuning()
        
    def _setup_fine_tuning(self):
        """Gèle les premières couches, dégèle les dernières"""
        # Nombre de couches à dégeler
        layers_to_unfreeze = 20
        
        # Geler toutes les couches d'abord
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Dégeler les dernières couches
        total_params = len(list(self.backbone.parameters()))
        for i, param in enumerate(self.backbone.parameters()):
            if i >= total_params - layers_to_unfreeze:
                param.requires_grad = True
        
        # Toujours dégeler la tête de classification
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)
    
    def get_trainable_params(self):
        """Retourne le nombre de paramètres entraînables"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total, trainable/total*100
    
    def print_architecture(self):
        """Affiche l'architecture du modèle"""
        print("="*60)
        print("ARCHITECTURE XCEPTIONNET")
        print("="*60)
        print("• Backbone: Xception pré-entraîné ImageNet")
        print("• Type: Depthwise Separable Convolutions")
        print("• Couches: 71 couches convolutionnelles")
        print("• Activation: ReLU")
        print("• Pooling: MaxPool2d + Global Average Pooling")
        print("• Tête: Linear(2048→512→2) avec Dropout")
        print("• Fine-tuning: 20 dernières couches dégelées")
        print("="*60)
        
        trainable, total, percentage = self.get_trainable_params()
        print(f"\n Paramètres: {trainable:,}/{total:,} ({percentage:.1f}% entraînables)")