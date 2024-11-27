import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from model.residual import ResidualStack
from vector_quantize_pytorch import VectorQuantize
class DisEncoding(nn.Module):
    def __init__(self, fea_emb, num_emb, fea_num, h_dim, n_res_layers):
        super(DisEncoding, self).__init__()

        self.fea_emb = fea_emb
        self.num_emb = num_emb
        self.fea_num = fea_num
        self.h_dim = h_dim
        self.n_res_layers = n_res_layers

        self.encoder = nn.Sequential(
            nn.Linear(1, self.h_dim // 4),
            nn.LayerNorm(normalized_shape=[self.h_dim // 4], elementwise_affine=True),
            nn.Tanh(),
            nn.Linear(self.h_dim // 4, self.h_dim // 2),
            nn.LayerNorm(normalized_shape=[self.h_dim // 2], elementwise_affine=True),
            nn.Tanh(),
            nn.Linear(self.h_dim // 2, self.h_dim),
            ResidualStack(self.h_dim, self.h_dim, self.h_dim//2, self.n_res_layers),
        )

        self.vqs = torch.nn.ModuleList([
            VectorQuantize(dim=self.h_dim, codebook_dim=self.num_emb, codebook_size=self.num_emb) for _ in range(self.fea_num)
        ])

        self.decoder = nn.Sequential(
            ResidualStack(self.h_dim, self.h_dim, self.h_dim // 2, self.n_res_layers),
            nn.Linear(self.h_dim, self.h_dim // 2),
            nn.LayerNorm(normalized_shape=[self.h_dim // 2], elementwise_affine=True),
            nn.Tanh(),
            nn.Linear(self.h_dim // 2, self.h_dim // 4),
            nn.LayerNorm(normalized_shape=[self.h_dim // 4], elementwise_affine=True),
            nn.Tanh(),
            nn.Linear(self.h_dim // 4, 1),
        )

    def forward(self, x):
        batch_size, seq_len = x.shape
        # (batch, seq ,1)
        inputs = x.unsqueeze(-1)
        # (batch, seq , emb)
        inputs = self.encoder(inputs)
        quantized_encoded_data = torch.zeros_like(inputs)
        commit_losses = []
        one_hot_encoded_data = torch.zeros((inputs.shape[0], inputs.shape[1], self.num_emb), dtype=torch.float32)
        for i in range(seq_len):
            current_position_vectors = inputs[:, i:i + 1, :]
            vq = self.vqs[i]

            quantized_current_position, indices, commit_loss = vq(current_position_vectors)
            index_one_hot = torch.nn.functional.one_hot(indices.squeeze(1), num_classes=self.num_emb).float()
            one_hot_encoded_data[:, i, :] = index_one_hot
            quantized_encoded_data[:, i, :] = quantized_current_position.squeeze(1)
            commit_losses.append(commit_loss)

        total_commit_loss = sum(commit_losses) / seq_len

        output = self.decoder(quantized_encoded_data)
        output = output.view([batch_size, -1])

        recons_loss = F.mse_loss(output, x)
        loss_1 = recons_loss + total_commit_loss
        return output, loss_1, one_hot_encoded_data

    def loss_function(self, output, input, vq_loss):

        recons_loss = F.mse_loss(output, input)

        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss': vq_loss}


if __name__ == "__main__":
    pass

