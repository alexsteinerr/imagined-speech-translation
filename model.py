import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel as HFViTModel
import math, numpy as np, scipy.signal

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, feature_size, num_layers=2, dropout=0.4):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, feature_size)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        outputs, _ = self.lstm(x)
        attn_weights = F.softmax(self.attention(outputs), dim=1)
        weighted_sum = (outputs * attn_weights).sum(dim=1)
        weighted_sum = self.dropout(weighted_sum)
        feat = self.fc(weighted_sum)
        return feat

class ViTEncoder(nn.Module):
    def __init__(self, feature_size, freeze_backbone=True):
        super(ViTEncoder, self).__init__()
        self.vit = HFViTModel.from_pretrained("google/vit-large-patch16-224-in21k")
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, feature_size)
        )
    def forward(self, x):
        outputs = self.vit(x)
        cls_token = outputs.last_hidden_state[:, 0, :]
        feat = self.fc(cls_token)
        return feat

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x, A):
        out = torch.bmm(A, x)
        out = self.linear(out)
        return out

class GNNEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=2):
        super(GNNEncoder, self).__init__()
        layers = []
        layers.append(GCNLayer(in_features, hidden_features))

        for _ in range(num_layers - 2):
            layers.append(GCNLayer(hidden_features, hidden_features))
        layers.append(GCNLayer(hidden_features, out_features))
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x, A):
        for layer in self.layers:
            x = layer(x, A)
            x = F.relu(x)
        x = x.mean(dim=1)

        return x

def compute_plv(eeg_tensor):
    batch, channels, time = eeg_tensor.shape
    plv_matrices = []
    for i in range(batch):
        sample = eeg_tensor[i].cpu().numpy()
        analytic = scipy.signal.hilbert(sample, axis=1)
        phase = np.angle(analytic)
        plv_matrix = np.zeros((channels, channels))

        for ch1 in range(channels):
            for ch2 in range(channels):
                phase_diff = phase[ch1] - phase[ch2]
                plv_matrix[ch1, ch2] = np.abs(np.mean(np.exp(1j * phase_diff)))

        plv_tensor = torch.tensor(plv_matrix, dtype=torch.float32, device=eeg_tensor.device)
        plv_matrices.append(plv_tensor.unsqueeze(0))

    plv = torch.cat(plv_matrices, dim=0)

    return plv

class EEG2TextEnsemble(nn.Module):
    def __init__(self, vocab_size, feature_size=768):
        super(EEG2TextEnsemble, self).__init__()
        self.vocab_size = vocab_size
        self.lstm_encoder = LSTMEncoder(input_size=125, hidden_size=128, feature_size=feature_size)
        self.vit_encoder = ViTEncoder(feature_size=feature_size, freeze_backbone=True)
        self.gnn_encoder = GNNEncoder(in_features=1, hidden_features=16, out_features=feature_size, num_layers=2)
        
        self.lstm_fc = nn.Linear(feature_size, vocab_size)
        self.vit_fc = nn.Linear(feature_size, vocab_size)
        self.gnn_fc = nn.Linear(feature_size, vocab_size)
        
        self.fusion_fc = nn.Sequential(
            nn.Linear(3 * feature_size, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, vocab_size)
        )

        self.ensemble_weights = nn.Parameter(torch.ones(4))

    def forward(self, eeg_lstm, spectrogram, eeg_for_gnn):
        lstm_feat = self.lstm_encoder(eeg_lstm)
        logits_lstm = self.lstm_fc(lstm_feat)
        vit_feat = self.vit_encoder(spectrogram)
        logits_vit = self.vit_fc(vit_feat)
        node_features = eeg_for_gnn.mean(dim=-1, keepdim=True)
        plv = compute_plv(eeg_for_gnn)

        gnn_feat = self.gnn_encoder(node_features, plv)
        logits_gnn = self.gnn_fc(gnn_feat)
        fusion_input = torch.cat([lstm_feat, vit_feat, gnn_feat], dim=-1)
        logits_fusion = self.fusion_fc(fusion_input)
        weights = torch.softmax(self.ensemble_weights, dim=0)
        final_logits = (weights[0] * logits_lstm + weights[1] * logits_vit + weights[2] * logits_gnn + weights[3] * logits_fusion)
        
        return final_logits, logits_lstm, logits_vit, logits_gnn, logits_fusion