import copy
import warnings
warnings.filterwarnings("ignore")
import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
import torch.nn.functional as F

# class LoRALinear(nn.Module):
#     def __init__(self, in_features, out_features, r=4, alpha=1, dropout=0.0, bias=True):
#         super(LoRALinear, self).__init__()
#         self.linear = nn.Linear(in_features, out_features, bias=bias)
#         self.lora_A = nn.Linear(in_features, r, bias=False)
#         self.lora_B = nn.Linear(r, out_features, bias=False)
#         self.alpha = alpha
#         self.dropout = nn.Dropout(dropout)
#
#         nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
#         nn.init.zeros_(self.lora_B.weight)
#
#     def forward(self, x):
#         return self.linear(x) + self.alpha * self.lora_B(self.dropout(self.lora_A(x)))


# def full_block(in_features, out_features, p_drop=0.1, use_lora=False, lora_r=4, lora_alpha=1, lora_dropout=0.0):
#     if use_lora:
#         linear = LoRALinear(in_features, out_features, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, bias=True)
#     else:
#         linear = nn.Linear(in_features, out_features, bias=True)
#     return nn.Sequential(
#         linear,
#         nn.LayerNorm(out_features),
#         nn.GELU(),
#         nn.Dropout(p=p_drop),
#     )


def full_block(in_features, out_features, p_drop=0.1):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.LayerNorm(out_features),
        nn.GELU(),
        nn.Dropout(p=p_drop),
    )


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1536):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp \
            (torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, token_dim: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, output_dim: int, dropout: float = 0.05):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.d_model = d_model

        self.encoder = nn.Sequential(nn.Linear(token_dim, d_model),
                                     nn.GELU(),
                                     nn.LayerNorm(d_model))

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.d_model = d_model
        self.dropout = dropout

        self.decoder = nn.Sequential(full_block(d_model, 1024, self.dropout),
                                     full_block(1024, output_dim, self.dropout),
                                     full_block(output_dim, output_dim, self.dropout),
                                     nn.Linear(output_dim, output_dim)
                                     )

        self.binary_decoder = nn.Sequential(
            full_block(output_dim + 1280, 2048, self.dropout),
            full_block(2048, 512, self.dropout),
            full_block(512, 128, self.dropout),
            nn.Linear(128, 1)
        )

        self.gene_embedding_layer = nn.Sequential(nn.Linear(token_dim, d_model),
                                                  nn.GELU(),
                                                  nn.LayerNorm(d_model))

        self.pe_embedding = None

    def forward(self, src: Tensor, mask: Tensor):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=(1 -mask))
        gene_output = self.decoder(output) # batch x seq_len x 128
        # embedding = torch.mul(gene_output, mask.t().unsqueeze(2)).sum(0) # average over non zero genes
        # In the new format, the cls token, which is at the 0 index mark, is the output.
        embedding = gene_output[0, :, :] # select only the CLS token.
        embedding = nn.functional.normalize(embedding, dim=1) # Normalize.
        return gene_output, embedding

    def predict(self, cell_embedding, gene_embeddings):
        gene_embeddings = self.gene_embedding_layer(gene_embeddings)
        dec = self.binary_decoder\
            (torch.hstack((cell_embedding, gene_embeddings)))
        return dec


class UniCure(nn.Module):
    def __init__(self, uce_lora: nn.Module = None,
                 drug_window_size: int = 32, drug_slide_step: int = 16,
                 cell_window_size: int = 32, cell_slide_step: int = 16,
                 hidden_dim: int = 64, output_size: int = 978, dropout_rate: float = 0.0):
        super(UniCure, self).__init__()

        # assert isinstance(uce_lora, nn.Module), "uce_lora must be an instance of nn.Module"

        self.uce_lora = uce_lora
        self.drug_window_size = drug_window_size
        self.drug_slide_step = drug_slide_step
        self.cell_window_size = cell_window_size
        self.cell_slide_step = cell_slide_step

        # bulk
        self.query_map = nn.Linear(self.cell_window_size, hidden_dim)  # Q cell
        self.key_map = nn.Linear(self.drug_window_size, hidden_dim)    # K drug
        self.value_map = nn.Linear(self.drug_window_size, hidden_dim)  # V drug

        self.dropout = nn.Dropout(p=dropout_rate)
        self.cell_windows_num = math.floor((1280 - cell_window_size) / cell_slide_step + 1)

        self.fusion_decoder = nn.Sequential(
            nn.Linear(hidden_dim * self.cell_windows_num, hidden_dim * self.cell_windows_num),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * self.cell_windows_num, hidden_dim * self.cell_windows_num),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * self.cell_windows_num, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, output_size),
            # nn.ReLU(),
            # nn.Dropout(dropout_rate),
            # nn.Linear(output_size, output_size),
        )

        # self._initialize_weights()

    def _initialize_weights(self):
        """
        Custom weights initialization using kaiming_normal_
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):  # Only apply to Linear layers
                # Initialize weights using kaiming_normal_
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:  # Initialize biases to zero
                    nn.init.zeros_(m.bias)

    def baseline_forward(self, cell_src: Tensor, mask: Tensor):
        cell_src = cell_src.permute(1, 0)
        cell_src = self.uce_lora.module.pe_embedding(cell_src.long())
        cell_src = nn.functional.normalize(cell_src, dim=2)

        _, cell_emb = self.uce_lora(cell_src, mask)
        cell_sw_embed = torch.nn.functional.unfold(
            cell_emb.unsqueeze(1).unsqueeze(-2), kernel_size=(1, self.cell_window_size), stride=(1, self.cell_slide_step)
        ).permute(0, 2, 1)
        Q = self.query_map(cell_sw_embed)
        Q = self.dropout(Q)
        Q_flat = Q.flatten(start_dim=1)
        output = self.decoder(Q_flat)
        return output

    def baseline_without_uce_forward(self, cell_emb: Tensor):
        cell_sw_embed = torch.nn.functional.unfold(
            cell_emb.unsqueeze(1).unsqueeze(-2), kernel_size=(1, self.cell_window_size), stride=(1, self.cell_slide_step)
        ).permute(0, 2, 1)
        Q = self.query_map(cell_sw_embed)
        Q = self.dropout(Q)
        Q_flat = Q.flatten(start_dim=1)
        output = self.decoder(Q_flat)
        return output

    def generate_uce_lora_emb(self, cell_src: Tensor, mask: Tensor):
        cell_src = cell_src.permute(1, 0)
        cell_src = self.uce_lora.module.pe_embedding(cell_src.long())
        cell_src = nn.functional.normalize(cell_src, dim=2)

        _, cell_emb = self.uce_lora(cell_src, mask)
        return cell_emb

    def pertrub_forward(self, cell_emb: Tensor, drug_emb: Tensor):
        cell_sw_embed = torch.nn.functional.unfold(
            cell_emb.unsqueeze(1).unsqueeze(-2), kernel_size=(1, self.cell_window_size), stride=(1, self.cell_slide_step)
        ).permute(0, 2, 1)
        drug_sw_embed = torch.nn.functional.unfold(
            drug_emb.unsqueeze(1).unsqueeze(-2), kernel_size=(1, self.drug_window_size), stride=(1, self.drug_slide_step)
        ).permute(0, 2, 1)

        # drug_sw_embed: (batch, num_windows, drug_window_size)
        Q = self.query_map(cell_sw_embed)  # shape (batch, num_windows, hidden_dim)
        K = self.key_map(drug_sw_embed)    # shape (batch, num_windows, hidden_dim)
        V = self.value_map(drug_sw_embed)  # shape (batch, num_windows, hidden_dim)

        # Apply dropout after projections
        Q = self.dropout(Q)
        K = self.dropout(K)
        V = self.dropout(V)

        # Cross-Attention
        attn_weights = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / K.size(-1)**0.5, dim=-1)
        fusion_output = torch.bmm(attn_weights, V)  # shape (batch, num_windows, hidden_dim)
        fusion_output_flat = fusion_output.flatten(start_dim=1)
        fusion_output = self.fusion_decoder(fusion_output_flat)
        # output
        final_output = self.decoder(fusion_output)
        return final_output

    def forward(self, mode, *args, **kwargs):
        if mode == "baseline":
            return self.baseline_forward(*args, **kwargs)
        elif mode == "generate_emb":
            return self.generate_uce_lora_emb(*args, **kwargs)
        elif mode == "baseline_without_uce":
            return self.baseline_without_uce_forward(*args, **kwargs)
        else:
            return self.pertrub_forward(*args, **kwargs)


class UniCureFTsc(nn.Module):
    def __init__(self, pretrained_model, hidden_dim: int = 2048, output_size: int = 1923, dropout_rate: float = 0.2):
        super(UniCureFTsc, self).__init__()
        self.pretrained_model = pretrained_model

        self.adaptor = nn.Sequential(
            # nn.Linear(output_size, output_size),
            nn.Linear(978, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_size),
        )

    def forward(self, channel, cell, drug):   #  alpha: Coefficient [0, 1] range, 0 = closer to cell line or less fine-tuning data, 1 = closer to in vivo and more fine-tuning data.
        out = self.pretrained_model(channel, cell, drug)  # Pass through the pre-trained model
        out = self.adaptor(out)
        return out


class UniCureFTtahoe(nn.Module):
    def __init__(self, pretrained_model, hidden_dim: int = 4096, output_size: int = 2990, dropout_rate: float = 0.2):
        super(UniCureFTtahoe, self).__init__()
        self.pretrained_model = pretrained_model

        self.adaptor = nn.Sequential(
            # nn.Linear(output_size, output_size),
            nn.Linear(1923, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_size),
        )

    def forward(self, channel, cell, drug):   #  alpha: Coefficient [0, 1] range, 0 = closer to cell line or less fine-tuning data, 1 = closer to in vivo and more fine-tuning data.
        out = self.pretrained_model(channel, cell, drug)  # Pass through the pre-trained model
        out = self.adaptor(out)
        return out


class UniCureFT2(nn.Module):
    def __init__(self, pretrained_model, hidden_dim: int = 1000, output_size: int = 978, dropout_rate: float = 0.2):
        super(UniCureFT2, self).__init__()
        self.pretrained_model = pretrained_model

        self.adaptor = nn.Sequential(
            # nn.Linear(output_size, output_size)
            nn.Linear(output_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_size),
        )

    def forward(self, cell, drug):   # alpha: Coefficient [0, 1] range, 0 = closer to cell line or less fine-tuning data, 1 = closer to in vivo and more fine-tuning data.
        out = self.pretrained_model("pertrub_forward", cell, drug)  # Pass through the pre-trained model
        out = self.adaptor(out)
        return out

class UniCureFT(nn.Module):
    def __init__(self, pretrained_model, hidden_dim: int = 1000, output_size: int = 978, dropout_rate: float = 0.2):
        super(UniCureFT, self).__init__()
        self.pretrained_model = pretrained_model

        self.adaptor = nn.Sequential(
            # nn.Linear(output_size, output_size)
            nn.Linear(output_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_size),
        )

    def forward(self, cell, drug, alpha=0.1):   # alpha: Coefficient [0, 1] range, 0 = closer to cell line or less fine-tuning data, 1 = closer to in vivo and more fine-tuning data.
        out = self.pretrained_model("pertrub_forward", cell, drug)  # Pass through the pre-trained model
        adapt = self.adaptor(out)
        out = out + (alpha * adapt)
        return out

# class BaselineCureALLWrapper:
#     def __init__(self, model):
#         self.model = model
#
#     def __call__(self, *args, **kwargs):
#         # baseline_forward
#         return self.model.module.baseline_forward(*args, **kwargs) \
#             if isinstance(self.model, nn.DataParallel) else self.model.baseline_forward(*args, **kwargs)

class baselineModel(nn.Module):
    def __init__(self, input_size=1792, hidden_size=2048, output_size=1000):
        super(baselineModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


#%% Transigen
class TranSiGen(torch.nn.Module):
    def __init__(self, n_genes, n_latent, n_en_hidden, n_de_hidden, features_dim, features_embed_dim, **kwargs):
        """ Constructor for class TranSiGen """
        super(TranSiGen, self).__init__()
        self.n_genes = n_genes
        self.n_latent = n_latent
        self.n_en_hidden = n_en_hidden
        self.n_de_hidden = n_de_hidden
        self.features_dim = features_dim
        self.feat_embed_dim = features_embed_dim
        self.init_w = kwargs.get('init_w', False)
        self.dev = kwargs.get('device', torch.device('cpu'))
        self.dropout = kwargs.get('dropout', 0.3)

        encoder = [
            nn.Linear(self.n_genes, self.n_en_hidden[0]),
            nn.BatchNorm1d(self.n_en_hidden[0]),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        ]
        if len(n_en_hidden) > 1:
            for i in range(len(n_en_hidden) - 1):
                en_hidden = [
                    nn.Linear(self.n_en_hidden[i], self.n_en_hidden[i + 1]),
                    nn.BatchNorm1d(self.n_en_hidden[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
                ]
                encoder = encoder + en_hidden

        self.encoder_x2 = nn.Sequential(*encoder)

        encoder_x1 = copy.deepcopy(encoder)
        self.encoder_x1 = nn.Sequential(*encoder_x1)

        self.mu_z2 = nn.Sequential(nn.Linear(self.n_en_hidden[-1], self.n_latent), )
        self.logvar_z2 = nn.Sequential(nn.Linear(self.n_en_hidden[-1], self.n_latent), )

        self.mu_z1 = nn.Sequential(nn.Linear(self.n_en_hidden[-1], self.n_latent), )
        self.logvar_z1 = nn.Sequential(nn.Linear(self.n_en_hidden[-1], self.n_latent), )

        if len(n_de_hidden) == 0:
            decoder = [nn.Linear(self.n_latent, self.n_genes)]
        else:
            decoder = [
                nn.Linear(self.n_latent, self.n_de_hidden[0]),
                nn.BatchNorm1d(self.n_de_hidden[0]),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ]

            if len(n_de_hidden) > 1:
                for i in range(len(self.n_de_hidden) - 1):
                    de_hidden = [
                        nn.Linear(self.n_de_hidden[i], self.n_de_hidden[i + 1]),
                        nn.BatchNorm1d(self.n_de_hidden[i + 1]),
                        nn.ReLU(),
                        nn.Dropout(self.dropout)
                    ]
                    decoder = decoder + de_hidden

            decoder.append(nn.Linear(self.n_de_hidden[-1], self.n_genes))
            decoder.append(nn.ReLU())

        self.decoder_x2 = nn.Sequential(*decoder)

        decoder_x1 = copy.deepcopy(decoder)
        self.decoder_x1 = nn.Sequential(*decoder_x1)

        if self.feat_embed_dim == None:
            self.mu_z2Fz1 = nn.Sequential(nn.Linear(self.n_latent + self.features_dim, self.n_latent), )
            self.logvar_z2Fz1 = nn.Sequential(nn.Linear(self.n_latent + self.features_dim, self.n_latent), )
        else:
            feat_embeddings = [
                nn.Linear(self.features_dim, self.feat_embed_dim[0]),
                nn.BatchNorm1d(self.feat_embed_dim[0]),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ]
            if len(self.feat_embed_dim) > 1:
                for i in range(len(self.feat_embed_dim) - 1):
                    feat_hidden = [
                        nn.Linear(self.feat_embed_dim[i], self.feat_embed_dim[i + 1]),
                        nn.BatchNorm1d(self.feat_embed_dim[i + 1]),
                        nn.ReLU(),
                        nn.Dropout(self.dropout)
                    ]
                    feat_embeddings = feat_embeddings + feat_hidden
            self.feat_embeddings = nn.Sequential(*feat_embeddings)

            self.mu_z2Fz1 = nn.Sequential(nn.Linear(self.n_latent + self.feat_embed_dim[-1], self.n_latent), )
            self.logvar_z2Fz1 = nn.Sequential(nn.Linear(self.n_latent + self.feat_embed_dim[-1], self.n_latent), )

        if self.init_w:
            self.encoder_x1.apply(self._init_weights)
            self.decoder_x1.apply(self._init_weights)

            self.encoder_x2.apply(self._init_weights)
            self.decoder_x2.apply(self._init_weights)

            self.mu_z1.apply(self._init_weights)
            self.logvar_z1.apply(self._init_weights)

            self.mu_z2.apply(self._init_weights)
            self.logvar_z2.apply(self._init_weights)

    def _init_weights(self, layer):
        """ Initialize weights of layer with Xavier uniform"""
        if type(layer) == nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight)
        return

    def encode_x1(self, X):
        """ Encode data """
        y = self.encoder_x1(X)
        mu, logvar = self.mu_z1(y), self.logvar_z1(y)
        z = self.sample_latent(mu, logvar)
        return z, mu, logvar

    def encode_x2(self, X):
        """ Encode data """
        y = self.encoder_x2(X)
        mu, logvar = self.mu_z2(y), self.logvar_z2(y)
        z = self.sample_latent(mu, logvar)
        return z, mu, logvar

    def decode_x1(self, z):
        """ Decode data """
        X_rec = self.decoder_x1(z)
        return X_rec

    def decode_x2(self, z):
        """ Decode data """
        X_rec = self.decoder_x2(z)
        return X_rec

    def sample_latent(self, mu, logvar):
        """ Sample latent space with reparametrization trick. First convert to std, sample normal(0,1) and get Z."""
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(self.dev)
        eps = eps.mul_(std).add_(mu)
        return eps

    def forward(self, x1, features):
        """ Forward pass through full network"""
        z1, mu1, logvar1 = self.encode_x1(x1)
        x1_rec = self.decode_x1(z1)

        if self.feat_embed_dim != None:
            feat_embed = self.feat_embeddings(features)
        else:
            feat_embed = features
        z1_feat = torch.cat([z1, feat_embed], 1)
        mu_pred, logvar_pred = self.mu_z2Fz1(z1_feat), self.logvar_z2Fz1(z1_feat)
        z2_pred = self.sample_latent(mu_pred, logvar_pred)
        x2_pred = self.decode_x2(z2_pred)

        return x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred

    def loss(self, x1_train, x1_rec, mu1, logvar1, x2_train, x2_rec, mu2, logvar2, x2_pred, mu_pred, logvar_pred):

        mse_x1 = F.mse_loss(x1_rec, x1_train, reduction="sum")
        mse_x2 = F.mse_loss(x2_rec, x2_train, reduction="sum")
        mse_pert = F.mse_loss(x2_pred - x1_train, x2_train - x1_train, reduction="sum")

        kld_x1 = -0.5 * torch.sum(1. + logvar1 - mu1.pow(2) - logvar1.exp(), )
        kld_x2 = -0.5 * torch.sum(1. + logvar2 - mu2.pow(2) - logvar2.exp(), )
        kld_pert = -0.5 * torch.sum(
            1 + (logvar_pred - logvar2) - ((mu_pred - mu2).pow(2) + logvar_pred.exp()) / logvar2.exp(), )

        return mse_x1 + mse_x2 + mse_pert + kld_x1 + kld_x2 + kld_pert, \
            mse_x1, mse_x2, mse_pert, kld_x1, kld_x2, kld_pert


