import copy
import warnings
warnings.filterwarnings("ignore")
import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
import torch.nn.functional as F


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

