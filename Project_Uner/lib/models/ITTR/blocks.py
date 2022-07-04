import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_chs):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_chs, in_chs, kernel_size=3, padding=1, groups=in_chs),
            nn.InstanceNorm2d(in_chs),
            nn.GELU()
        )

    def forward(self, x):
        return self.layer(x) + x


class HPB(nn.Module):
    def __init__(self, in_chs, patch_size=16, head_chs=64, num_head=8):
        super().__init__()
        self.dpsa = DPSA(in_chs, patch_size, head_chs, num_head)

        self.dwconv = nn.Conv2d(in_chs, in_chs, kernel_size=3, padding=1, groups=dim, bias=True)
        self.mlp = nn.Conv2d(in_chs*2, in_chs, kernel_size=1, bias=True)

        self.res_block = ResBlock(in_chs)
        self.conv_ffn = nn.Sequential(
            nn.LayerNorm(in_chs),
            nn.Conv2d(in_chs, in_chs, kernel_size=1),
            nn.InstanceNorm2d(in_chs),
            nn.GELU(),
            ResBlock(in_chs),
            nn.Conv2d(in_chs, in_chs, kernel_size=1),
            nn.InstanceNorm2d(in_chs),
        )

    def forward(self, x):
        global_branch = self.dpsa(x)
        local_branch = self.dwconv(x)

        f_branch = torch.cat((local_branch, global_branch), dim=1)
        f_branch = self.mlp(f_branch) + x
        return self.conv_ffn(f_branch)


class DPSA(nn.Module):
    def __init__(self, in_chs, patch_size=16, head_chs=64, num_head=8):
        super().__init__()
        self.patch_size = patch_size
        self.num_head = num_head
        self.head_chs = head_chs
        hid_chs = num_head * head_chs

        self.layer_norm = nn.LayerNorm(in_chs)
        self.query_conv = nn.Conv2d(in_chs, hid_chs, kernel_size=1, bias=False)
        self.key_conv = nn.Conv2d(in_chs, hid_chs, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(in_chs, hid_chs, kernel_size=1, bias=False)
        self.out_conv = nn.Conv2d(hid_chs, in_chs, kernel_size=1, bias=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.layer_norm(x)

        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        query = query.reshape(B, self.heads, -1, H, W).reshape(B * self.heads, -1, H, W)
        key = key.reshape(B, self.heads, -1, H, W).reshape(B * self.heads, -1, H, W)
        v = value.reshape(B, self.heads, -1, H, W).reshape(B * self.heads, -1, H, W)

        q = F.normalize(query, dim=1)
        k = F.normalize(key, dim=1)

        query_sum = torch.sum(q, dim=(1, 2))
        key_height = torch.sum(k, dim=2)
        key_width = torch.sum(k, dim=1)

        # row-wise sum
        h_index = (query_sum[:, None, :] * key_height).sum(dim=2).topk(k=self.patch_size, dim=-1).indices
        h_index = h_index[..., None, None].repeat((1, 1, k.shape[-2], self.head_chs))
        k = k.gather(1, h_index)
        v = v.gather(1, h_index)

        # col-wise sum
        w_index = (query_sum[:, None, :] * key_width).sum(dim=1).topk(k=self.patch_size, dim=-1).indices
        w_index = w_index[:, None, :, None].repeat((1, k.shape[1], 1, self.head_chs))
        k = k.gather(2, w_index)
        v = v.gather(2, w_index)

        q = q.reshape(B * self.heads, -1, W)
        k = k.reshape(B * self.heads, -1, W)
        v = v.reshape(B * self.heads, -1, W)

        # cosine similarities
        sim = (q[:, :, None, :] * k[:, None, :, :]).sum(dim=-1)

        # attention
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn[..., None] * v[:, None, :, :]).sum(dim=2)
        out = out.reshape(B, self.heads, H, W, -1).permute((0, 1, 4, 2, 3)).reshape(B, -1, H, W)

        return self.out_conv(out)