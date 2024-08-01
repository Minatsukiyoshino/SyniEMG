import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_lstm=False, use_attention=False):
        super(ConvBlock, self).__init__()
        self.use_lstm = use_lstm
        self.use_attention = use_attention

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm1d(out_channels)
        self.relu = nn.ReLU()

        if self.use_lstm:
            self.lstm = nn.LSTM(out_channels, out_channels, batch_first=True)

        if self.use_attention:
            self.attn = nn.MultiheadAttention(embed_dim=out_channels, num_heads=4, batch_first=True)

    def forward(self, x):
        x = self.relu(self.norm(self.conv(x)))

        if self.use_lstm:
            x = x.permute(0, 2, 1)
            x, _ = self.lstm(x)
            x = x.permute(0, 2, 1)

        if self.use_attention:
            x = x.permute(2, 0, 1)
            x, _ = self.attn(x, x, x)
            x = x.permute(1, 2, 0)

        return x


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_lstm=False, use_attention=False):
        super(UpConvBlock, self).__init__()
        self.use_lstm = use_lstm
        self.use_attention = use_attention

        self.upconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv1d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm1d(out_channels)
        self.relu = nn.ReLU()

        if self.use_lstm:
            self.lstm = nn.LSTM(out_channels, out_channels, batch_first=True)

        if self.use_attention:
            self.attn = nn.MultiheadAttention(embed_dim=out_channels, num_heads=4, batch_first=True)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat((x, skip), dim=1)
        x = self.relu(self.norm(self.conv(x)))

        if self.use_lstm:
            x = x.permute(0, 2, 1)
            x, _ = self.lstm(x)
            x = x.permute(0, 2, 1)

        if self.use_attention:
            x = x.permute(2, 0, 1)
            x, _ = self.attn(x, x, x)
            x = x.permute(1, 2, 0)

        return x


class UNet1D_LA(nn.Module):
    def __init__(self, input_dim=3, output_dim=4, hidden_dim=64, use_lstm=True, use_attention=False):
        super(UNet1D_LA, self).__init__()

        self.enc1 = ConvBlock(input_dim, hidden_dim, use_lstm, use_attention)
        self.enc2 = ConvBlock(hidden_dim, hidden_dim * 2, use_lstm, use_attention)
        self.enc3 = ConvBlock(hidden_dim * 2, hidden_dim * 4, use_lstm, use_attention)
        self.enc4 = ConvBlock(hidden_dim * 4, hidden_dim * 8, use_lstm, True)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.up1 = UpConvBlock(hidden_dim * 8, hidden_dim * 4, use_lstm, True)
        self.up2 = UpConvBlock(hidden_dim * 4, hidden_dim * 2, use_lstm, use_attention)
        self.up3 = UpConvBlock(hidden_dim * 2, hidden_dim, use_lstm, use_attention)

        self.dec4 = nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        dec1 = self.up1(enc4, enc3)
        dec2 = self.up2(dec1, enc2)
        dec3 = self.up3(dec2, enc1)
        dec4 = self.dec4(dec3)

        return dec4.permute(0, 2, 1)
