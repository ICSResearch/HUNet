from math import prod
import torch
import torch.nn as nn
from models.common.ops import (
    window_partition,
    window_reverse,
    to_2tuple,
)
from timm.models.layers import DropPath
from models.common.utils import (
    DMlp,
    trunc_normal_,
)

class PSA(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size# Wh, Ww
        self.permuted_window_size = (window_size[0] // 2,window_size[1] // 2)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.permuted_window_size[0] - 1) * (2 * self.permuted_window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise aligned relative position index for each token inside the window
        coords_h = torch.arange(self.permuted_window_size[0])
        coords_w = torch.arange(self.permuted_window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="xy"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.permuted_window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.permuted_window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.permuted_window_size[1] - 1
        aligned_relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        aligned_relative_position_index = aligned_relative_position_index.reshape\
            (self.permuted_window_size[0],self.permuted_window_size[1],1,1,self.permuted_window_size[0]*self.permuted_window_size[1]).repeat(1,1,2,2,1)\
            .permute(0,2,1,3,4).reshape(4*self.permuted_window_size[0]*self.permuted_window_size[1],self.permuted_window_size[0]*self.permuted_window_size[1]) #  FN*FN,WN*WN
        self.register_buffer('aligned_relative_position_index', aligned_relative_position_index)
        # compresses the channel dimension of KV
        self.kv = nn.Linear(dim, dim//2, bias=qkv_bias)
        self.q = nn.Linear(dim,dim,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape
        # compress the channel dimension of KV :(num_windows*b, num_heads, n//4, c//num_heads)
        kv = self.kv(x).reshape(b_,self.permuted_window_size[0],2,self.permuted_window_size[1],2,2,c//4).permute(0,1,3,5,2,4,6).reshape(b_, n//4, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # keep the channel dimension of Q: (num_windows*b, num_heads, n, c//num_heads)
        q = self.q(x).reshape(b_, n, 1, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))   # (num_windows*b, num_heads, n, n//4)

        relative_position_bias = self.relative_position_bias_table[self.aligned_relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.permuted_window_size[0] * self.permuted_window_size[1], -1)  # (n, n//4)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, n, n//4)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_//nw, nw, self.num_heads, n, n//4) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n//4)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.permuted_window_size}, num_heads={self.num_heads}'

    def flops(self, n):
        # calculate flops for 1 window with token length of n
        flops = 0
        # qkv = self.qkv(x)
        flops += n * self.dim * 1.5 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * n * (self.dim // self.num_heads) * n/4
        #  x = (attn @ v)
        flops += self.num_heads * n * n/4 * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += n * self.dim * self.dim
        return flops

class WindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        window_size,
        num_heads,
        window_shift=False,
        attn_drop=0.0,
        pretrained_window_size=[0, 0],
        args=None,
    ):

        super(WindowAttention, self).__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size[0]
        self.permuted_window_size = self.window_size//2
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.shift_size = window_size[0] // 2 if window_shift else 0
        self.euclidean_dist = args.euclidean_dist
        self.attn_psa = PSA(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=attn_drop,
            proj_drop=0.
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)
        
    def calculate_mask(self, x_size):
        # calculate mask for original windows
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1\
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, to_2tuple(self.window_size))  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        # calculate mask for permuted windows
        h, w = x_size
        permuted_window_mask = torch.zeros((1, h // 2, w // 2, 1))  # 1 h w 1
        h_slices = (slice(0, -self.permuted_window_size), slice(-self.permuted_window_size,
                                                            -self.shift_size // 2), slice(-self.shift_size // 2, None))
        w_slices = (slice(0, -self.permuted_window_size), slice(-self.permuted_window_size,
                                                            -self.shift_size // 2),
                    slice(-self.shift_size // 2, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                permuted_window_mask[:, h, w, :] = cnt
                cnt += 1

        permuted_windows = window_partition(permuted_window_mask, to_2tuple(self.permuted_window_size))
        permuted_windows = permuted_windows.view(-1, self.permuted_window_size * self.permuted_window_size)
        # calculate attention mask
        attn_mask = mask_windows.unsqueeze(2) - permuted_windows.unsqueeze(1)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask


    def forward(self, qkv, x_size):
        H, W = x_size
        B, L, C = qkv.shape
        qkv = qkv.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            qkv = torch.roll(
                qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )

        # partition windows
        qkv = window_partition(qkv, to_2tuple(self.window_size))  # nW*B, wh, ww, C
        qkv = qkv.view(-1, prod(to_2tuple(self.window_size)), C)  # nW*B, wh*ww, C

        attn_windows = self.attn_psa(qkv, self.attn_mask)

        x = window_reverse(attn_windows, to_2tuple(self.window_size), x_size)  # B, H, W, C/3

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(B, L, C)

        return x

    def extra_repr(self) -> str:
        return (
            f"window_size={self.window_size}, shift_size={self.shift_size}, "
            f"pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}"
        )

    def flops(self, N):
        pass


class PWB(nn.Module):
    def __init__(
        self,
        dim,
        up,
        num_heads_w,
        x_size,
        window_size=7,
        window_shift=False,
        mlp_ratio=4.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=[0, 0],
        args=None,
    ):
        super().__init__()
        self.dim = dim
        self.up = up
        self.num_heads_w = num_heads_w
        self.x_size = x_size
        self.window_size = window_size
        self.window_shift = window_shift
        self.args = args
        self.mlp_ratio = mlp_ratio

        self.attn = WindowAttention(
            dim,
            x_size,
            window_size,
            num_heads_w,
            window_shift,
            attn_drop,
            pretrained_window_size,
            args,
        )
        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = DMlp(dim=dim)
        self.norm2 = norm_layer(dim)
        self.pretrained_window_size=[0, 0]
        self.anchor_window_down_factor=1
        self.shift_size = 4

    def forward(self, x):
        # conv & attention
        x = x + self.drop_path(
            self.norm1(self.attn(x, self.x_size))
        )
        # x = x + self.drop_path(self.norm2(bchw_to_blc(self.mlp(blc_to_bchw(x,self.x_size)))))
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim},  num_heads=({self.num_heads_w}), "
            f"window_size={self.window_size}, window_shift={self.window_shift}. "
        )

    def flops(self):
        pass
