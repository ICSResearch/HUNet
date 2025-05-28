import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath
from omegaconf import OmegaConf
from models.common.mixed_attn_block_efficient import (
    PWB,
)
from models.common.utils import(
    Upsample,
    Downsample,
    IterAppend,
)

from models.common.ops import (
    bchw_to_blc,
    blc_to_bchw,
)
from timm.models.layers import to_2tuple, trunc_normal_
from models.common import config


class Head(nn.Module):
    def __init__(self, drop_path=0.1):
        r""" """
        super().__init__()
        self.embed_dim = config.para.embed_dim
        self.block = nn.Sequential(
            nn.Conv2d(1, self.embed_dim//3, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim//3, self.embed_dim//3, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim//3, self.embed_dim, 3, padding=1),
        )
        alpha_0 = 1e-2
        self.alpha = nn.Parameter(
            alpha_0 * torch.ones((1, self.embed_dim, 1, 1)), requires_grad=True
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.alpha * self.block(x))
        return x


class Tail(nn.Module):
    def __init__(self):
        r""" """
        super().__init__()
        self.embed_dim = config.para.embed_dim
        self.block = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim, self.embed_dim//3, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim//3, self.embed_dim//3, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim//3, 1, 3, padding=1),
        )
    def forward(self, x):
        x = self.block(x)
        return x

class TransformerStage(nn.Module):

    def __init__(
        self,
        no,
        dim,
        depth,
        patch_size,
        num_heads_window,
        window_size,
        mlp_ratio=4.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=[0, 0],
        init_method="",
        args=None,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.init_method = init_method
        self.lambda_ = nn.Parameter(torch.Tensor([0.5]))
        self.gamma = nn.Parameter(torch.full((1, 1, dim * 2 ** (depth // 2 - 1)), 0.1**(no+1)))
        self.patch_embed = Head()
        self.conv_last = Tail()
        self.encoder = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.up_sample = nn.ModuleList()
        for i in range(depth):
            self.encoder.append(PWB(
                dim=dim * 2 ** (i//2),
                up = 0,
                num_heads_w=num_heads_window,
                x_size = (patch_size // 2**(i//2), patch_size // 2**(i//2)),
                window_size=window_size,
                window_shift=i % 2 == 1,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
                args=args,
            ) 
        )
            if i % 2 == 1 and i != (depth-1):
                self.down_sample.append(Downsample(dim * 2 ** (i//2), (patch_size // 2**(i//2), patch_size // 2**(i//2))))
        for i in range(depth):
            self.decoder.append(
                PWB(
                dim=dim * 2 ** (i//2),
                up = 1,
                num_heads_w=num_heads_window,
                x_size = (patch_size // 2**(i//2), patch_size // 2**(i//2)),
                window_size=window_size,
                window_shift=i % 2 == 1,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
                args=args,
                )
            )
            if i % 2 == 1 and i != 1:
                self.up_sample.append(Upsample(dim * 2 ** (i//2), (patch_size // 2**(i//2), patch_size // 2**(i//2))))
        self.up_sample = self.up_sample[::-1]
        self.decoder = self.decoder[::-1]

    def _init_weights(self):
        for n, m in self.named_modules():
            if self.init_method == "w":
                if isinstance(m, (nn.Linear, nn.Conv2d)) and n.find("cpb_mlp") < 0:
                    print("nn.Linear and nn.Conv2d weight initilization")
                    m.weight.data *= 0.1
            elif self.init_method == "l":
                if isinstance(m, nn.LayerNorm):
                    print("nn.LayerNorm initialization")
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 0)
            elif self.init_method.find("t") >= 0:
                scale = 0.1 ** (len(self.init_method) - 1) * int(self.init_method[-1])
                if isinstance(m, nn.Linear) and n.find("cpb_mlp") < 0:
                    trunc_normal_(m.weight, std=scale)
                elif isinstance(m, nn.Conv2d):
                    m.weight.data *= 0.1
                print(
                    "Initialization nn.Linear - trunc_normal; nn.Conv2d - weight rescale."
                )
            else:
                raise NotImplementedError(
                    f"Parameter initialization method {self.init_method} not implemented in TransformerStage."
                )

    def forward(self, x, x_size, y, Phiweight, soft_thr):
        
        res = x
        x = blc_to_bchw(x, x_size)
        r = x - self.lambda_ * PhiTPhi_fun(x, Phiweight) + self.lambda_ * y
        x = self.patch_embed(r)
        x = bchw_to_blc(x)
        x_ms = []
        i = 0
        cnt = 0 
        for encoder in self.encoder:
            x = encoder(x)
            x_ms.append(x)
            if i % 2 == 1 and i != (self.depth-1):
                x = self.down_sample[cnt](x)
                cnt += 1
            i += 1
        x = 0
        x_ms.reverse()
        i = 0
        cnt = 0 
        for x_e, decoder in zip(x_ms, self.decoder):
            if i == 0:
                B = x_e.shape[0]
                x_e = torch.mul(torch.sign(x_e), F.relu(torch.abs(x_e) - torch.tile(self.gamma, (B, 1, 1))*soft_thr[:, np.newaxis, np.newaxis]))
                
            x = decoder(x + x_e)
            if i % 2 == 1 and i != (self.depth-1):
                x = self.up_sample[cnt](x)
                cnt += 1
            i += 1
        x_n = x
        x = bchw_to_blc(self.conv_last(blc_to_bchw(x, x_size)))
        return x + res, x_n

    def flops(self):
        pass


def PhiTPhi_fun(x, A):
    temp = F.conv2d(x, A, padding=0,stride=config.para.patch_size, bias=None)
    temp = F.conv_transpose2d(temp, A, stride=config.para.patch_size)
    return temp


class HUNet(nn.Module):
    def __init__(
        self,
        patch_size=config.para.patch_size,
        in_channels=1,
        out_channels=None,
        embed_dim=config.para.embed_dim,
        depths=[6, 6, 6, 6, 6, 6, 6],
        num_heads_window=[3, 3, 3, 3, 3, 3, 3],
        window_size=8,
        mlp_ratio=4.0,
        out_proj_type="linear",
        local_connection=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=[0, 0],
        init_method="n",  # initialization method of the weight parameters used to train large scale models.
        euclidean_dist=False,
        **kwargs,
    ):
        super(HUNet, self).__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.layer_num = len(depths)
        self.n_input = int(config.para.rate * patch_size * patch_size)
        self.Phiweight = nn.Parameter(init.xavier_normal_(torch.Tensor(self.n_input, 1, self.patch_size, self.patch_size)))
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.Tensor([0.2]))
        self.window_size = to_2tuple(window_size)
        self.shift_size = [w // 2 for w in self.window_size]
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        args = OmegaConf.create(
            {
                "out_proj_type": out_proj_type,
                "local_connection": local_connection,
                "euclidean_dist": euclidean_dist,
            }
        )
        
        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            layer = TransformerStage(
                no = i, 
                dim=embed_dim,
                depth=depths[i],
                patch_size=self.patch_size,
                num_heads_window=num_heads_window[i],
                window_size=self.window_size,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i]) : sum(depths[: i + 1])
                ], 
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
                init_method=init_method,
                args=args,
            )
            self.layers.append(layer)

        self.merge = IterAppend(len(depths), embed_dim)
        self.conv_last = Tail()
        
        self.apply(self._init_weights)
        if init_method in ["l", "w"] or init_method.find("t") >= 0:
            for layer in self.layers:
                layer._init_weights()



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):

            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x, y, Phiweight, soft_thr):
        
        x_size = (x.shape[2], x.shape[3])
        x = bchw_to_blc(x)
        x = self.pos_drop(x)
        x_out = torch.tensor([]).to(device=config.para.device)
        x_muti = 0
        for layer in self.layers:
            x, x_n = layer(x, x_size, y, Phiweight, soft_thr)
            x_out = torch.cat([x_out, x], dim=-1)
            x_muti = x_muti + x_n
        x_out = blc_to_bchw(x_out, x_size)
        x_muti = blc_to_bchw(x_muti, x_size)
        x_out = self.merge(x_out)
        x_out = self.conv_last(x_out + self.weight * x_muti)
        return x_out

    def forward(self, inputs):
        PhiTb = F.conv2d(inputs, self.Phiweight, stride=self.patch_size, padding=0, bias=None)  
        y = F.conv_transpose2d(PhiTb, self.Phiweight, stride=self.patch_size) 
        x = y
        B = x.shape[0]
        soft_thr = x.abs().view(B, -1).max(dim=1)[0]
        x = self.forward_features(x, y, self.Phiweight, soft_thr)
        return x

    def flops(self):
        pass

    
