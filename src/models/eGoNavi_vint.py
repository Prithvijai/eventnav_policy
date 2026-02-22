import torch
import torch.nn as nn


class eGoNavi_ViNT(nn.Module):
    def __init__(self, token_dim=512, num_tokens=6, num_layers=4, num_heads=4, ff_dim=2048):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, token_dim))
        # print(self.pos_embed.shape)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True
        )
        # print(encoder_layer)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # print(self.transformer)

    def forward(self, tokens):
        # tokens: (B, 6, 512)
        x = tokens + self.pos_embed
        x = self.transformer(x)
        # Return the "current" observation token (index 4 if 0-based, else adjust)
        return x[:, 4, :]
    

transformer = eGoNavi_ViNT()