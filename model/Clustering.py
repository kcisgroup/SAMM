import torch
import torch.nn as nn
from torch.nn import functional as F
from model.Mining import Mining
from vector_quantize_pytorch import VectorQuantize
from model.residual import ResidualStack
class Clustering(nn.Module):
    def __init__(self, args):
        super(Clustering, self).__init__()
        self.fea_emb = args.fea_emb_dim
        self.num_emb = args.symbol_space
        self.fea_num = args.feature_dim
        self.cluster_num = args.cluster_space
        self.word_emb_num = args.word_emb_dim
        self.lm_path = args.lm_path
        self.h_dim = args.hidden_dim
        self.n_res_layers = args.n_res_layers
        self.device = torch.device(f'cuda:{args.gpu_id}' if args.cuda else 'cpu')

        self.Mining = Mining(self.fea_emb, self.num_emb, self.fea_num, self.lm_path, self.h_dim, self.n_res_layers, self.device)


        self.encoder = nn.Sequential(
            nn.Linear(self.word_emb_num * (self.fea_num + 2), self.word_emb_num),
            nn.BatchNorm1d(self.word_emb_num),
            nn.Tanh(),
            nn.Linear(self.word_emb_num, self.word_emb_num // 2),
            nn.BatchNorm1d(self.word_emb_num // 2),
            nn.Tanh(),
            nn.Linear(self.word_emb_num // 2, self.word_emb_num // 4),
            nn.BatchNorm1d(self.word_emb_num // 4),
            ResidualStack(self.word_emb_num//4, self.word_emb_num//4, self.word_emb_num, 1),
        )

        self.vq = VectorQuantize(dim=self.word_emb_num // 4, codebook_dim=self.cluster_num, codebook_size=self.cluster_num)

        self.decoder = nn.Sequential(
            ResidualStack(self.word_emb_num // 4, self.word_emb_num // 4, self.word_emb_num, 1),
            nn.Linear(self.word_emb_num // 4, self.word_emb_num // 2),
            nn.BatchNorm1d(self.word_emb_num // 2),
            nn.Tanh(),
            nn.Linear(self.word_emb_num // 2, self.word_emb_num),
            nn.BatchNorm1d(self.word_emb_num),
            nn.Tanh(),
            nn.Linear(self.word_emb_num, self.word_emb_num * (self.fea_num+2)),
            # nn.BatchNorm1d(self.word_emb_num * (self.fea_num+2)),
        )


    def forward(self, x):
        encoding_one_hot, recons_loss, mlm_loss, outputs, sen_ids, cls_vectors = self.Mining(x)
        encoded_vector = self.encoder(cls_vectors)
        quantized, indices, commit_loss_2 = self.vq(encoded_vector)
        clusters = indices
        decoded_vector = self.decoder(quantized)
        recons_loss_2 = F.mse_loss(decoded_vector, cls_vectors)
        all_loss = recons_loss + mlm_loss  + recons_loss_2 + commit_loss_2
        return sen_ids, all_loss, clusters



if __name__ == "__main__":
    pass

