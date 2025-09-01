import torch

from utils import save_image_pred, save_json_pred, colorize_target_image

def calculate_accuracy(nearest_patch_indices, seg_colors_src, seg_colors_tgt, seg_sizes_tgt):

    device = seg_colors_src.device
    BACKGROUND_COLOR = torch.tensor([1, 1, 1, 0], dtype=torch.uint8, device=device)
    
    total_seg_num = 0
    correct_seg_num = 0
    total_pix_num = 0
    correct_pix_num = 0
    
    PIXEL_THRES = 10
    total_seg_thres_num = 0
    correct_seg_thres_num = 0
    
    total_fg_pix_num = 0
    correct_fg_pix_num = 0
    
    pix_bg_miou_numerator = 0 # True Positive
    pix_bg_miou_denominator = 0 # True Positive + False Positive + False Negative
    
    seg_bg_miou_numerator = 0 # True Positive
    seg_bg_miou_denominator = 0 # True Positive + False Positive + False Negative


    for i, color_index in enumerate(nearest_patch_indices):
        
        gt_color = seg_colors_tgt[i]
        pred_color = seg_colors_src[color_index]
        seg_size = seg_sizes_tgt[i]
        
        total_pix_num += seg_size
        total_seg_num += 1
        
        is_bg_gt = torch.equal(gt_color, BACKGROUND_COLOR)
        is_bg_pred = torch.equal(pred_color, BACKGROUND_COLOR)
        
        if PIXEL_THRES < seg_size:
            total_seg_thres_num += 1
        if not is_bg_gt:
            total_fg_pix_num += seg_size
            
            
        if torch.equal(pred_color, gt_color):
            correct_pix_num += seg_size
            correct_seg_num += 1
            if PIXEL_THRES < seg_size:
                correct_seg_thres_num += 1
            if is_bg_gt:
                pix_bg_miou_numerator += seg_size # True Positive
                pix_bg_miou_denominator += seg_size
                seg_bg_miou_numerator += 1
                seg_bg_miou_denominator += 1               
            else:
                correct_fg_pix_num += seg_size

        else:
            if is_bg_gt or is_bg_pred:
                pix_bg_miou_denominator += seg_size # False Negative, False Positive
                seg_bg_miou_denominator += 1


    pix_acc = correct_pix_num / total_pix_num if total_pix_num > 0 else 0
    seg_acc = correct_seg_num / total_seg_num if total_seg_num > 0 else 0
    seg_acc_thres = correct_seg_thres_num / total_seg_thres_num if total_seg_thres_num > 0 else 0
    
    pix_fg_acc = correct_fg_pix_num / total_fg_pix_num if total_fg_pix_num > 0 else 0
    pix_bg_acc = pix_bg_miou_numerator / pix_bg_miou_denominator if pix_bg_miou_denominator > 0 else 0
    seg_bg_miou =  seg_bg_miou_numerator / seg_bg_miou_denominator if seg_bg_miou_denominator > 0 else 0
    
    return seg_acc, seg_acc_thres, pix_acc, pix_fg_acc, pix_bg_acc, seg_bg_miou


def calculate_metrics(seg_sim_map, data, save_images, save_json, save_path):
    
    seg_colors_src = data['seg_colors_src']
    seg_colors_tgt = data['seg_colors_tgt']
    char_name = data['char_name']
    frame_indices_tgt = data['frame_indices_tgt']
    line_images_tgt = data['line_images_tgt']
    seg_images_tgt = data['seg_images_tgt']
    seg_sizes_tgt = data['seg_sizes_tgt']
    
    B, S_tgt, _, _ = seg_colors_tgt.shape
    
    total_seg_acc = 0.0
    total_seg_acc_thres = 0.0
    total_pix_acc = 0.0

    total_pix_fg_acc = 0.0
    total_pix_bg_miou = 0.0
    total_seg_bg_miou = 0.0

    metrics_data_num = 0
    
    for b in range(B):

        seg_colors_src_batch = seg_colors_src[b]
        S_src, L_src, _ = seg_colors_src_batch.shape
        seg_colors_src_batch = seg_colors_src_batch.view(S_src*L_src, -1)
        padding_color = torch.tensor([-1., -1., -1., -1.], device=seg_colors_src_batch.device)
        valid_mask_src = ~torch.all(seg_colors_src_batch == padding_color, dim=1)
        seg_colors_src_batch = seg_colors_src_batch[valid_mask_src]

        for s in range(S_tgt):
            
            seg_colors_tgt_batch = seg_colors_tgt[b][s]
            L_tgt, _ = seg_colors_tgt_batch.shape
            valid_mask_tgt = ~torch.all(seg_colors_tgt_batch == padding_color, dim=1)
            seg_colors_tgt_batch = seg_colors_tgt_batch[valid_mask_tgt]
            
            seg_sim_map_batch = seg_sim_map[b, s*L_tgt:(s+1)*L_tgt, :]

            seg_sim_map_batch = seg_sim_map_batch[valid_mask_tgt][:, valid_mask_src] 
            nearest_patch_indices = torch.argmax(seg_sim_map_batch, dim=-1)
            
            seg_color_list_src = seg_colors_src_batch
            color_list_pred = seg_colors_src_batch[nearest_patch_indices]
            color_list_pred = color_list_pred * 255

            seg_color_list_gt = seg_colors_tgt_batch
            seg_sizes_list_tgt = seg_sizes_tgt[b][s]
            seg_sizes_list_tgt = seg_sizes_list_tgt[valid_mask_tgt]

            seg_acc, seg_acc_thres, pix_acc, pix_fg_acc, pix_bg_miou, seg_bg_miou = calculate_accuracy(nearest_patch_indices, seg_color_list_src, seg_color_list_gt, seg_sizes_list_tgt)

            if save_images:
                image_pred = colorize_target_image(color_list_pred, line_images_tgt[b][s], seg_images_tgt[b][s])
                save_image_pred(image_pred, char_name[b], frame_indices_tgt[b][s], save_path)

            if save_json:
                save_json_pred(color_list_pred, char_name[b], frame_indices_tgt[b][s], save_path)

            total_seg_acc += seg_acc
            total_seg_acc_thres += seg_acc_thres
            total_pix_acc += pix_acc

            total_pix_fg_acc += pix_fg_acc
            total_pix_bg_miou += pix_bg_miou
            total_seg_bg_miou += seg_bg_miou

            metrics_data_num += 1
  
        metrics = {
        'total_seg_acc': total_seg_acc,
        'total_seg_acc_thres': total_seg_acc_thres,
        'total_pix_acc': total_pix_acc,
        'total_pix_fg_acc': total_pix_fg_acc,
        'total_pix_bg_miou': total_pix_bg_miou,
        'total_seg_bg_miou': total_seg_bg_miou,
        'metrics_data_num': metrics_data_num
        }

    return metrics


def calculate_metrics_multi_ref(data, seg_sim_map, all_seg_colors_ref, save_images, save_json, save_path):
    
    char_name = data["char_name"]
    frame_name = data["frame_name"]
    line_image_tgt = data["line_image"] 
    seg_image_tgt = data["seg_image"] 
    seg_colors_tgt = data["seg_colors"]
    seg_sizes_tgt = data["seg_sizes"]
    B, _, _ = seg_colors_tgt.shape
    
    total_seg_acc = 0.0
    total_seg_acc_thres = 0.0
    total_pix_acc = 0.0
    total_pix_fg_acc = 0.0
    total_pix_bg_miou = 0.0
    total_seg_bg_miou = 0.0

    metrics_data_num = 0
    
    for b in range(B):

        seg_color_list_gt = seg_colors_tgt[b]
        seg_sim_map_batch = seg_sim_map[b]
        nearest_patch_indices = torch.argmax(seg_sim_map_batch, dim=-1)
        
        color_list_pred = all_seg_colors_ref[nearest_patch_indices]
        color_list_pred = color_list_pred * 255

        seg_sizes_list_tgt = seg_sizes_tgt[b]

        seg_acc, seg_acc_thres, pix_acc, pix_fg_acc, pix_bg_miou, seg_bg_miou = calculate_accuracy(nearest_patch_indices, all_seg_colors_ref, seg_color_list_gt, seg_sizes_list_tgt)

        if save_images:
            image_pred = colorize_target_image(color_list_pred, line_image_tgt[b], seg_image_tgt[b])
            save_image_pred(image_pred, char_name[b], frame_name[b], save_path)

        if save_json:
            save_json_pred(color_list_pred, char_name[b], frame_name[b], save_path)

        total_seg_acc += seg_acc
        total_seg_acc_thres += seg_acc_thres
        total_pix_acc += pix_acc

        total_pix_fg_acc += pix_fg_acc
        total_pix_bg_miou += pix_bg_miou
        total_seg_bg_miou += seg_bg_miou

        metrics_data_num += 1
  
        metrics = {
        'total_seg_acc': total_seg_acc,
        'total_seg_acc_thres': total_seg_acc_thres,
        'total_pix_acc': total_pix_acc,
        'total_pix_fg_acc': total_pix_fg_acc,
        'total_pix_bg_miou': total_pix_bg_miou,
        'total_seg_bg_miou': total_seg_bg_miou,
        'metrics_data_num': metrics_data_num
        }

    return metrics
