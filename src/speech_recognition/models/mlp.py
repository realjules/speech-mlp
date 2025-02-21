import torch
import torch.nn as nn
import torch.nn.init as init

class SpeechMLP(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def __init__(self, input_size, output_size, dropout_rate=0.05):
        super().__init__()

        self.model = nn.Sequential(
            # First Linear Layer
            nn.Linear(input_size, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            # Second Linear Layer
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            # Third Linear Layer
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            # Fourth Linear Layer
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            # Fifth Linear Layer
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            # Sixth Linear Layer
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            # Seventh Linear Layer
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            # Output Layer
            nn.Linear(256, output_size)
        )

        self._initialize_weights()

    def forward(self, x):
        return self.model(x)