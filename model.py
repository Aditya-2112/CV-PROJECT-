import torch.nn as nn

class TrafficSignCNN(nn.Module):
    """
    A custom Convolutional Neural Network for GTSRB traffic sign classification.
    Expects input shape: (Batch, 3, 32, 32)
    Output shape: (Batch, 43)
    """
    def __init__(self, num_classes=43):
        super(TrafficSignCNN, self).__init__()
        
        # Block 1: Extract low-level features (edges, basic colors)
        # Input: 3x32x32 -> Output: 32x16x16
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), # Stabilizes training by normalizing feature maps
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: Mid-level texture features
        # Input: 32x16x16 -> Output: 64x8x8
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: Higher-level shape patterns
        # Input: 64x8x8 -> Output: 128x4x4
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 4: Compact global features
        # Input: 128x4x4 -> Output: 256x1x1
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # AdaptiveAvgPool2d(1) dynamically averages spatial dimensions to 1x1,
            # making the model robust to slight variations and removing spatial dims cleanly.
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Fully Connected Layer for classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Dropout random zeroes out 50% of connections to prevent overfitting
            nn.Dropout(p=0.5), 
            nn.Linear(in_features=256, out_features=num_classes)
        )

    def forward(self, x):
        """Forward pass defining how data flows through the network."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x
