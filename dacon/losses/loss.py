import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

    def preprocess_colors(self, data, b):

        seg_colors_src = data['seg_colors_src'][b]
        seg_colors_tgt = data['seg_colors_tgt'][b]
 
        S_src, L_src, C = seg_colors_src.shape
        flatten_seg_color_src = seg_colors_src.view(S_src*L_src, C)

        S_tgt, L_tgt, C = seg_colors_tgt.shape
        flatten_seg_color_tgt = seg_colors_tgt.view(S_tgt*L_tgt, C)

        padding_color = torch.tensor([-1.0, -1.0, -1.0, -1.0], device=seg_colors_src.device)
        valid_mask_src = ~torch.all(flatten_seg_color_src == padding_color, dim=1)
        valid_mask_tgt = ~torch.all(flatten_seg_color_tgt == padding_color, dim=1)

        flatten_seg_color_src = flatten_seg_color_src[valid_mask_src]
        unique_color_src = torch.unique(flatten_seg_color_src, dim=0)

        matches = torch.any(
            torch.all(unique_color_src[:, None, :] == flatten_seg_color_tgt[None, :, :], dim=2), dim=0
        )
        padding_color_expanded = padding_color.unsqueeze(0).expand_as(flatten_seg_color_tgt)
        flatten_seg_color_tgt = torch.where(matches[:, None], flatten_seg_color_tgt, padding_color_expanded)

        if not torch.any((unique_color_src == padding_color).all(dim=1)):
            unique_color_src = torch.cat([unique_color_src, padding_color.unsqueeze(0)], dim=0)
        padding_index = unique_color_src.size(0) - 1  


        seg_colors_tgt = flatten_seg_color_tgt.view(S_tgt, L_tgt, C)
        valid_mask_tgt = valid_mask_tgt.view(S_tgt, L_tgt)


        return flatten_seg_color_src, seg_colors_tgt, unique_color_src, valid_mask_src, valid_mask_tgt, padding_index
    

class CrossEntropyLoss(BaseLoss):
    def __init__(self,):
        super(CrossEntropyLoss, self).__init__()
        self.tau = 1e-1
        self.epsilon = 1e-15
        
    def forward(self, seg_sim_map, data):
        B = seg_sim_map.shape[0]
        device = seg_sim_map.device
        total_loss = torch.tensor(0.0, device=device)
        loss_data_num = 0
        
        for b in range(B):
            color_list_src, color_list_gt, unique_color_list, valid_mask_src, valid_mask_tgt, padding_index = \
                self.preprocess_colors(data, b)
            S_tgt, L_tgt, _ = color_list_gt.shape

            match_src = (color_list_src.unsqueeze(1) == unique_color_list.unsqueeze(0)).all(dim=2)
            color_indices = match_src.float().argmax(dim=1) 

            for s in range(S_tgt):
                if valid_mask_tgt.sum() == 0:
                    continue  

                seg_sim_map_batch = seg_sim_map[b, s*L_tgt:(s+1)*L_tgt, :]
                seg_sim_map_batch = seg_sim_map_batch[valid_mask_tgt[s]][:, valid_mask_src]

                color_list_gt_batch = color_list_gt[s][valid_mask_tgt[s]]

                match_gt = (color_list_gt_batch.unsqueeze(1) == unique_color_list.unsqueeze(0)).all(dim=2)
                gt_color_indices = match_gt.float().argmax(dim=1)

                scaled_similarities = seg_sim_map_batch / self.tau
                softmax_similarities = F.softmax(scaled_similarities, dim=-1)

                pred_color_index_pcts = torch.zeros(
                    color_list_gt_batch.size(0), unique_color_list.size(0),
                    dtype=torch.float, device=device
                )
                pred_color_index_pcts.index_add_(1, color_indices, softmax_similarities)

                pred_logits = torch.log(pred_color_index_pcts + self.epsilon)
                valid_mask = gt_color_indices != padding_index
    
                loss = F.nll_loss(pred_logits[valid_mask], gt_color_indices[valid_mask])
                
                total_loss += loss
                loss_data_num += 1
            
        return total_loss / loss_data_num if loss_data_num > 0 else total_loss, loss_data_num

    
class MAELoss(BaseLoss):
    def __init__(self):
        super(MAELoss, self).__init__()
        self.tau = 1e-1

    def forward(self, seg_sim_map, dino_sim_map, data):
        B = seg_sim_map.shape[0]
        device = seg_sim_map.device
        total_loss = torch.zeros(1, device=device)
        loss_data_num = 0

        for b in range(B):
            _, color_list_gt, _, _, valid_mask_tgt, _ = self.preprocess_colors(data, b) 
            S_tgt, L_tgt, _ = color_list_gt.shape

            for s in range(S_tgt):
                if valid_mask_tgt[s].sum() == 0:
                    continue  
    
                seg_sim_map_batch = seg_sim_map[b, s*L_tgt:(s+1)*L_tgt, :]
                dino_sim_map_batch = dino_sim_map[b, s*L_tgt:(s+1)*L_tgt, :]

                seg_sim_map_batch = seg_sim_map_batch / self.tau
                dino_sim_map_batch = dino_sim_map_batch / self.tau

                loss = F.l1_loss(dino_sim_map_batch, dino_sim_map_batch)
                total_loss += loss 
                loss_data_num += 1
            
        return total_loss / loss_data_num if loss_data_num > 0 else total_loss, loss_data_num
    


class CombinedLoss(nn.Module):
    def __init__(self, loss_scale_ce=0.5, loss_scale_mae=0.2):
        super(CombinedLoss, self).__init__()
        self.ce_loss = CrossEntropyLoss()
        self.mae_loss = MAELoss()
        self.loss_scale_ce = loss_scale_ce
        self.loss_scale_mae = loss_scale_mae

    def forward(self, data, seg_sim_map, dino_seg_sim_map):
        ce_loss, _ = self.ce_loss(seg_sim_map, data)
        
        mae_loss, _ = self.mae_loss(seg_sim_map, dino_seg_sim_map, data)

        total_loss = self.loss_scale_ce * ce_loss + self.loss_scale_mae * mae_loss
        
        return total_loss, ce_loss, mae_loss


