import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import segment_pooling

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
    
        return x

class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), 
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)

class UNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_list=[64, 128, 256, 512]):
        super(UNet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim_list = hidden_dim_list
        self.num_down_blocks = len(hidden_dim_list)

        self.pool = nn.MaxPool2d(2, stride=2)

        self.encoder_blocks = nn.ModuleList()
        current_input_dim = input_dim
        for current_output_dim in hidden_dim_list:
            self.encoder_blocks.append(ConvBlock(current_input_dim, current_output_dim))
            current_input_dim = current_output_dim

        bottleneck_input_dim = hidden_dim_list[-1]
        bottleneck_output_dim = bottleneck_input_dim * 2
        self.bottleneck = ConvBlock(bottleneck_input_dim, bottleneck_output_dim)
        
        self.decoder_up_convs = nn.ModuleList()
        self.decoder_conv_blocks = nn.ModuleList()

        current_decoder_input_dim = bottleneck_output_dim 
        
        for i in reversed(range(self.num_down_blocks)):
            skip_connection_dim = hidden_dim_list[i]
            upconv_output_dim = skip_connection_dim 
            self.decoder_up_convs.append(UpConv(current_decoder_input_dim, upconv_output_dim))
            self.decoder_conv_blocks.append(ConvBlock(upconv_output_dim * 2, upconv_output_dim))
            current_decoder_input_dim = upconv_output_dim 
            
        self.final_conv = nn.Conv2d(current_decoder_input_dim, output_dim, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x) 
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections_reversed = skip_connections[::-1]
        
        for i in range(self.num_down_blocks):
            skip_connection = skip_connections_reversed[i] 
            
            x = self.decoder_up_convs[i](x)
            x = torch.cat([skip_connection, x], dim=1) 
            x = self.decoder_conv_blocks[i](x)

        x = self.final_conv(x)
        return x


class DACoNModel(nn.Module):
    def __init__(self, dacon_config, version):
        super(DACoNModel, self).__init__()

        self.version = version
        model_type = dacon_config["dino_model_type"]

        try:
            print(f"Loading DINOv2 model '{model_type}' from PyTorch Hub...")
            self.dino = torch.hub.load('facebookresearch/dinov2', model_type)
            print(f"DINOv2 model '{model_type}' loaded successfully.")
        except Exception as e:
            raise ImportError(
                f"Failed to load DINOv2 model '{model_type}' from PyTorch Hub. "
                f"Please check your internet connection or ensure the model exists. Error: {e}"
            )
        for param in self.dino.parameters():
            param.requires_grad = False

        
        self.dino_dim = self.dino.embed_dim
        self.feats_dim = dacon_config["feats_dim"] 

        self.unet_input_size = dacon_config["unet_input_size"] 
        self.dino_input_size = dacon_config["dino_input_size"] 
        self.segment_pool_size = dacon_config["segment_pool_size"] 
        self.unet_hidden_dim_list = dacon_config["unet_hidden_dim_list"]

        self.unet = UNet(input_dim=3, output_dim=self.feats_dim, hidden_dim_list=self.unet_hidden_dim_list)
        self.dino_mlp = MLP(self.dino_dim, self.feats_dim*4, self.feats_dim)
        if self.version == "1_0":
            self.dino_unet_mlp = MLP(self.feats_dim * 2, self.feats_dim * 4, self.feats_dim)

    def l2_normalize(self, x, dim=-1, eps=1e-6):
        return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)
    
    def _prepare_images(self, images, target_size):
        images = images[:, :, 0:3, :, :]
        B, S, C, H, W = images.shape
        images = images.view(B * S, C, H, W)
        return F.interpolate(images, size=target_size, mode='bilinear', align_corners=False)

    def get_dino_feats_map(self, images):

        B, S, C, H, W = images.shape
        images = self._prepare_images(images, self.dino_input_size)
        dino_output = self.dino.get_intermediate_layers(images, n=1, return_class_token=False)
        patch_tokens = dino_output[0]
        feat_H = self.dino_input_size[0] // 14
        feat_W = self.dino_input_size[1] // 14

        return patch_tokens.permute(0, 2, 1).view(B, S, self.dino_dim, feat_H, feat_W)
        
       
    def get_unet_feats_map(self, images):
        B, S, C, H, W = images.shape
        images = self._prepare_images(images, self.unet_input_size)
        unet_outputs = self.unet(images)
        _, C, H, W = unet_outputs.shape

        return unet_outputs.view(B, S, C, H, W)

    def dino_dim_reduction(self, seg_dino_feats):
        B, S, L, C = seg_dino_feats.shape
        seg_dino_feats = self.dino_mlp(seg_dino_feats.view(B*S*L, C))

        return seg_dino_feats.view(B, S, L, -1)
    
    def dino_unet_fusion(self, seg_dino_feats, seg_unet_feats):
        B, S, L, C = seg_dino_feats.shape
        dino_flat_feats = seg_dino_feats.view(B*S*L, C)
        unet_flat_feats = seg_unet_feats.view(B*S*L, C)

        if self.version == "1_0":
            seg_fusion_feats = self.dino_unet_mlp(torch.cat([dino_flat_feats, unet_flat_feats], dim=-1))
        else:
            seg_fusion_feats = self.l2_normalize(dino_flat_feats) + self.l2_normalize(unet_flat_feats)
            
        return seg_fusion_feats.view(B, S, L, -1)

    def get_segment_feats(self, feats_map, seg_image, seg_num):

        if self.version == "1_0":
            _, _, H, W = seg_image.shape
            seg_feats_src = segment_pooling(feats_map, seg_image, seg_num, (H,W))
        else:
            seg_feats_src = segment_pooling(feats_map, seg_image, seg_num, self.segment_pool_size)
 
        return  seg_feats_src


    def get_seg_cos_sim(self, seg_feats_src, seg_feats_tgt):
        
        seg_feats_src = F.normalize(seg_feats_src, p=2, dim=-1)
        seg_feats_tgt = F.normalize(seg_feats_tgt, p=2, dim=-1)

        B, S_src, L_src, C = seg_feats_src.shape
        seg_feats_src = seg_feats_src.view(B, S_src*L_src, C)

        B, S_tgt, L_tgt, C = seg_feats_tgt.shape
        seg_feats_tgt = seg_feats_tgt.view(B, S_tgt*L_tgt, C)
        
        seg_cos_sim = torch.matmul(seg_feats_tgt, seg_feats_src.transpose(-1, -2))

        return seg_cos_sim
    
    def _process_single(self, line_image, seg_image, seg_num):

        line_images = line_image.unsqueeze(1)
        seg_images = seg_image.unsqueeze(1)
        seg_nums = seg_num.unsqueeze(1)

        seg_feats, raw_dino_feats = self._process_multi(line_images, seg_images, seg_nums)

        return seg_feats.squeeze(1), raw_dino_feats.squeeze(1)
    
    def _process_multi(self, line_images, seg_images, seg_nums):

        dino_feats_map = self.get_dino_feats_map(line_images)
        seg_dino_feats = self.get_segment_feats(dino_feats_map, seg_images, seg_nums)
        raw_dino_feats = seg_dino_feats.clone() 

        unet_feats_map = self.get_unet_feats_map(line_images)
        seg_unet_feats = self.get_segment_feats(unet_feats_map, seg_images, seg_nums)

        seg_dino_feats_reduced = self.dino_dim_reduction(seg_dino_feats)
        seg_feats = self.dino_unet_fusion(seg_dino_feats_reduced, seg_unet_feats)

        return seg_feats, raw_dino_feats

    def forward(self, data):

        seg_feats_src, seg_dino_feats_src = self._process_multi(data['line_images_src'], data['seg_images_src'], data["seg_nums_src"])
        seg_feats_tgt, seg_dino_feats_tgt = self._process_multi(data['line_images_tgt'], data['seg_images_tgt'], data["seg_nums_tgt"])
        dino_seg_cos_sim = self.get_seg_cos_sim(seg_dino_feats_src, seg_dino_feats_tgt)
        seg_cos_sim = self.get_seg_cos_sim(seg_feats_src, seg_feats_tgt)

        return seg_cos_sim, dino_seg_cos_sim



