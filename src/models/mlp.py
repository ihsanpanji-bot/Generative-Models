import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    """Classifier 784 -> 256 -> 64 -> 10 (struktur identik dg encoder AE)."""

    def __init__(self):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(784, 256), nn.Linear(256, 64), nn.Linear(64, 10)

    def features(self, x):
        """z_MLP in R^64: representasi sebelum layer klasifikasi akhir."""
        return F.relu(self.fc2(F.relu(self.fc1(x.flatten(1)))))

    def forward(self, x):
        return self.fc3(self.features(x))
