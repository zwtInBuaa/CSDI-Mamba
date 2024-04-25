import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from layers.S4Layer import S4Layer
from layers.Attention import *


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_CSDI_best(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]
        # add length of MTS

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    config,
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, config, side_dim, channels, diffusion_embedding_dim, nheads):
        self.length = config["eval_length"]
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        # self.linear_layer = nn.Linear(128, 64)

        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.cond_projection_1 = Conv1d_with_init(side_dim, channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.s4_init_layer = S4Layer(features=channels, lmax=100)

        # self.time_layer = EncoderLayer(
        #     d_time=32,
        #     d_feature=72,
        #     d_model=channels,
        #     d_inner=64,
        #     n_head=nheads,
        #     d_k=64,
        #     d_v=64,
        #     dropout=0.1,
        #     attn_dropout=0,
        # )
        # self.feature_layer = EncoderLayer(
        #     d_time=72,
        #     d_feature=32,
        #     d_model=channels,
        #     d_inner=64,
        #     n_head=nheads,
        #     d_k=64,
        #     d_v=64,
        #     dropout=0.1,
        #     attn_dropout=0,
        # )

        # self.feature_layer = EncoderLayer(
        #     d_time=32,
        #     actual_d_feature=72,
        #     d_model=channels,
        #     d_inner=64,
        #     n_head=nheads,
        #     d_k=64,
        #     d_v=64,
        #     dropout=0.1,
        #     attn_dropout=0,
        # )

        # self.w_tf = nn.Linear(2 * channels, channels)

        # self.transformer_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    # def forward_transformer(self, y, base_shape):
    #     # print(base_shape)
    #     B, channel, K, L = base_shape
    #     if L == 1 or K == 1:
    #         return y
    #     y = y.reshape(B, channel, K, L).permute(2, 3, 0, 1).reshape(K * L, B, channel)
    #     y = self.transformer_layer(y).permute(1, 2, 0)
    #     y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
    #     return y

    def forward_imputation(self, y, base_shape):
        B, channel, K, L = base_shape
        # enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # x = x.reshape(B, channel, K, L).permute(1, 0, 2, 3)
        # y = torch.zeros(channel, B, K, L).cuda()
        # for i in range(channel):
        #     y[i], attns = self.transformer_layer(x[i], attn_mask=None)
        # y = y.permute(1, 0, 2, 3).reshape(B, channel, K * L)

        # x, attns = self.transformer_layer(x.permute(2, 3, 0, 1), attn_mask=None)
        # x = x.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        # dec_out = self.projection(enc_out)
        y = y.reshape(B, channel, K, L).reshape(B, channel, K * L)
        y, attens = self.transformer_layer(y.permute(2, 0, 1))
        y = y.permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1))
        y = y.permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1))
        y = y.permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward_attention_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y, attens = self.time_layer(y.permute(0, 2, 1))
        y = y.permute(0, 2, 1)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_attention_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y, attens = self.feature_layer(y.permute(0, 2, 1))
        y = y.permute(0, 2, 1)
        y = y.reshape(B, K, channel, L).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        # y = self.forward_attention(y, base_shape)

        y = self.s4_init_layer(y.permute(2, 0, 1)).permute(1, 2, 0)

        O_t_time = self.forward_time(y, base_shape)
        O_t_feature = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        #
        # # y = self.forward_time(y, base_shape)
        # # y = self.forward_feature(y, base_shape)
        #
        # # # # print("y1:")
        # # # # print(y, y.shape)
        #
        # # method 1
        # # y = (O_t_time + O_t_feature) / 2
        # # method 2
        y = torch.sigmoid(O_t_time) * torch.tanh(O_t_feature)
        # method 3

        # O_t_time = O_t_time.permute(2, 0, 1)
        # O_t_feature = O_t_feature.permute(2, 0, 1)
        # O_t = self.w_tf(torch.cat((O_t_time, O_t_feature), dim=-1))
        # y = O_t.permute(1, 2, 0)

        # y1 = self.forward_time(y, base_shape)
        # y2 = self.forward_feature(y, base_shape)
        # y = self.forward_combined((y1+y2)/2,base_shape)
        # y = self.forward_combined(y, base_shape)
        # print("y2:")
        # print("ResidualBlock.forward y.shape", y.shape)
        # y = self.forward_imputation(y, base_shape)
        # print(y, y.shape)
        # y = self.forward_transformer(y, base_shape)
        # y = self.forward_feature(y, base_shape)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)
        # cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        # y = self.mid_projection(y)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip