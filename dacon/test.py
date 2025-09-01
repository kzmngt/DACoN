
import os
import sys
import yaml
import time
import torch
import argparse
import datetime
from pathlib import Path
from torch.utils.data import DataLoader

from models import DACoNModel
from data import DACoNDataset, DACoNSingleDataset, dacon_pad_collate_fn, dacon_single_pad_collate_fn
from losses import CombinedLoss
from utils import (
    make_val_data_list,
    move_data_to_device,
    load_config,
    format_time,
    calculate_metrics,
    calculate_metrics_multi_ref,
    setup_logger,
    get_folder_names,
    make_single_data_list,
    get_file_count,
)



def validate(model, val_dataloader, criterion, device, save_images, save_json, save_path):
    model.eval()

    total_loss = 0.0
    total_seg_acc = 0.0
    total_pix_acc = 0.0
    total_seg_acc_thres = 0.0
    total_pix_fg_acc = 0.0
    total_pix_bg_miou = 0.0
    total_seg_bg_miou = 0.0

    metrics_data_num = 0
    num_val_batches = len(val_dataloader)

    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            data = move_data_to_device(data, device)

            seg_sim_map, dino_seg_sim_map = model.forward(data)
            loss, _, _ = criterion(data, seg_sim_map, dino_seg_sim_map)
            loss = loss.item()
     
            metrics  = calculate_metrics(seg_sim_map, data, save_images, save_json, save_path)
            total_loss += loss
            total_seg_acc += metrics["total_seg_acc"]
            total_pix_acc += metrics["total_pix_acc"]
            total_seg_acc_thres += metrics["total_seg_acc_thres"]
            total_pix_fg_acc += metrics["total_pix_fg_acc"]
            total_pix_bg_miou += metrics["total_pix_bg_miou"]
            total_seg_bg_miou += metrics["total_seg_bg_miou"]
            metrics_data_num += metrics["metrics_data_num"]

            print(f"  Batch {i+1}/{num_val_batches} - Current Loss: {loss:.4f}", end='\r')
            

    loss = total_loss / metrics_data_num
    seg_acc = total_seg_acc / metrics_data_num
    seg_acc_thres = total_seg_acc_thres / metrics_data_num
    pix_acc = total_pix_acc / metrics_data_num
    pix_fg_acc = total_pix_fg_acc / metrics_data_num
    pix_bg_miou = total_pix_bg_miou / metrics_data_num
    seg_bg_miou = total_seg_bg_miou / metrics_data_num

    val_results = {
        "loss": loss,
        "seg_acc": seg_acc,
        "seg_acc_thres": seg_acc_thres,
        "pix_acc": pix_acc,
        "pix_fg_acc": pix_fg_acc,
        "pix_bg_miou": pix_bg_miou,
        "seg_bg_miou": seg_bg_miou
    }

    return val_results

def validate_multi_ref(model, config, device, save_images, save_json, save_path):
    model.eval()

    total_loss = 0.0
    total_seg_acc = 0.0
    total_pix_acc = 0.0
    total_seg_acc_thres = 0.0
    total_pix_fg_acc = 0.0
    total_pix_bg_miou = 0.0
    total_seg_bg_miou = 0.0

    metrics_data_num = 0

    ref_shot = config['ref_shot']
    val_data_root = config['datasets']['val']['root']
    val_batch_size = config['val']['batch_size']
    val_num_workers = config['datasets']['val']['num_worker']

    char_names = get_folder_names(val_data_root)

    val_data_num = 0
    for char_name in char_names:
        val_data_num += get_file_count(os.path.join(val_data_root, char_name, "gt"))

    sample_idx = 0

    with torch.no_grad():
        for char_name in char_names:
            ref_data_list = make_single_data_list(val_data_root, char_name, ref_shot, is_ref=True)
            ref_dataset = DACoNSingleDataset(ref_data_list, val_data_root, is_ref=True, mode = "val_kf")
            ref_dataloader = DataLoader(ref_dataset, batch_size=val_batch_size, shuffle=False, num_workers=val_num_workers, collate_fn=dacon_single_pad_collate_fn)

            all_seg_feats_ref = torch.empty(0, device=device)
            all_seg_colors_ref = torch.empty(0, device=device)

            for i, ref_data in enumerate(ref_dataloader):
                ref_data = move_data_to_device(ref_data, device)
                seg_colors_ref = ref_data['seg_colors']
                seg_feats_ref, _ = model._process_single(ref_data['line_image'], ref_data['seg_image'], ref_data["seg_num"])

                for b in range(val_batch_size):
                    all_seg_feats_ref = torch.cat((all_seg_feats_ref, seg_feats_ref[b]), dim = 0)
                    all_seg_colors_ref = torch.cat((all_seg_colors_ref, seg_colors_ref[b]), dim = 0)
                        
                del ref_data, seg_colors_ref, seg_feats_ref
                torch.cuda.empty_cache()

            val_data_list = make_single_data_list(val_data_root, char_name, ref_shot, is_ref=False)
            val_dataset = DACoNSingleDataset(val_data_list, val_data_root, is_ref=False, mode = "val_kf")
            val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=val_num_workers, collate_fn=dacon_single_pad_collate_fn)

            all_seg_feats_ref = all_seg_feats_ref.unsqueeze(0)
            all_seg_feats_ref = all_seg_feats_ref.repeat(val_batch_size, 1, 1)

            for i, data in enumerate(val_dataloader):
                data = move_data_to_device(data, device)
                seg_feats_tgt, _ = model._process_single(data['line_image'], data['seg_image'], data["seg_num"])
                seg_sim_map = model.get_seg_cos_sim(all_seg_feats_ref.unsqueeze(1), seg_feats_tgt.unsqueeze(1))
                seg_sim_map = seg_sim_map.squeeze(1)

                metrics  = calculate_metrics_multi_ref(data, seg_sim_map, all_seg_colors_ref, save_images, save_json, save_path)

                total_seg_acc += metrics["total_seg_acc"]
                total_pix_acc += metrics["total_pix_acc"]
                total_seg_acc_thres += metrics["total_seg_acc_thres"]
                total_pix_fg_acc += metrics["total_pix_fg_acc"]
                total_pix_bg_miou += metrics["total_pix_bg_miou"]
                total_seg_bg_miou += metrics["total_seg_bg_miou"]
                metrics_data_num += metrics["metrics_data_num"]

                sample_idx += 1
                print(f"  Sample {sample_idx}/{val_data_num}", end='\r')

                del data, seg_feats_tgt, seg_sim_map
                
            del all_seg_feats_ref, all_seg_colors_ref
            torch.cuda.empty_cache()

    loss = total_loss / metrics_data_num
    seg_acc = total_seg_acc / metrics_data_num
    seg_acc_thres = total_seg_acc_thres / metrics_data_num
    pix_acc = total_pix_acc / metrics_data_num
    pix_fg_acc = total_pix_fg_acc / metrics_data_num
    pix_bg_miou = total_pix_bg_miou / metrics_data_num
    seg_bg_miou = total_seg_bg_miou / metrics_data_num

    val_results = {
        "loss": loss,
        "seg_acc": seg_acc,
        "seg_acc_thres": seg_acc_thres,
        "pix_acc": pix_acc,
        "pix_fg_acc": pix_fg_acc,
        "pix_bg_miou": pix_bg_miou,
        "seg_bg_miou": seg_bg_miou
    }

    return val_results



def main(args): 
    config = load_config(args.config)
    model_path = args.model

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Hyperparameters from config ---
    version = args.version
    if version == None:
        version = config['version']

    colorize_type = config['colorize_type']
    ref_shot = config['ref_shot']

    val_data_root = config['datasets']['val']['root']
    num_workers_val = config['datasets']['val']['num_worker']
    clip_interval = config['datasets']['clip_interval']

    loss_scale_ce = config['losses']['loss_scale_ce']
    loss_scale_mae = config['losses']['loss_scale_mae']

    val_batch_size = config['val']['batch_size']
    save_images = config['val']['save_images']
    save_json = config['val']['save_json']
    save_path = config['val']['save_path']

    if colorize_type == "keyframe":
        save_path = os.path.join(save_path, "val", f"{colorize_type}_{ref_shot}")
    elif colorize_type == "consecutive_frame":
        save_path = os.path.join(save_path, "val", colorize_type)
    os.makedirs(save_path, exist_ok=True)

    logger = setup_logger(save_path, current_time, log_name="dacon_test")
    logger.info(f"Testing DACoN version {version}.")
    logger.info("\n===== Config =====\n" + yaml.dump(config, default_flow_style=False, sort_keys=False))

    device = torch.device("cuda" if torch.cuda.is_available() and config['num_gpu'] > 0 else "cpu")
    logger.info(f"Using device: {device}")

    model = DACoNModel(config['network'], version).to(device)
    logger.info("\n===== Model Architecture =====\n" + str(model))

    logger.info(f"Loading checkpoint {os.path.basename(model_path)}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # --- Dataset and DataLoader setup ---
    
    criterion = CombinedLoss(loss_scale_ce=loss_scale_ce, loss_scale_mae=loss_scale_mae).to(device)
    logger.info(f"Loss function initialized.")

    total_training_start_time = time.time()
    logger.info("--- Start Testing ---")

    if colorize_type == "keyframe":
        char_names = get_folder_names(val_data_root)
        val_data_num = 0
        for char_name in char_names:
            val_data_num += get_file_count(os.path.join(val_data_root, char_name, "gt"))
        logger.info(f"  Evaluating on {val_data_num} samples.")
        val_results = validate_multi_ref(model, config, device, save_images, save_json, save_path)

    elif colorize_type == "consecutive_frame":
        val_data_list = make_val_data_list(val_data_root, colorize_type, clip_interval, ref_shot)
        val_dataset = DACoNDataset(val_data_list, val_data_root, mode = "val_cf")
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers_val, collate_fn=dacon_pad_collate_fn)
        logger.info(f"  Evaluating on {len(val_dataset)} samples.")
        val_results = validate(model, val_dataloader, criterion, device, save_images, save_json, save_path)

    if colorize_type == "keyframe":
        logger.info(f"\n  Evaluation Results for KerFrame Colorization with Ref shot {ref_shot}:")
    elif colorize_type == "consecutive_frame":
        logger.info(f"\n  Evaluation Results for Consecutive Frame Colorization:")
    eval_summary = (
        f"\n  Average Loss: {val_results['loss']:.8f}\n"
        f"  Segment Accuracy: {val_results['seg_acc']:.4f} (Threshold: {val_results['seg_acc_thres']:.4f})\n"
        f"  Pixel Accuracy: {val_results['pix_acc']:.4f} (Foreground: {val_results['pix_fg_acc']:.4f})\n"
        f"  Pixel Background MIoU: {val_results['pix_bg_miou']:.4f}"
    )
    logger.info(eval_summary)

    del val_results
    torch.cuda.empty_cache()

    print("\n--- Test complete! ---")
    total_finish_time = time.time()
    print(f"\nTotal time: {format_time(total_finish_time - total_training_start_time)}")


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.append(str(ROOT))

    parser = argparse.ArgumentParser(description="Test the DACoN model.")
    parser.add_argument('--config', type=str,
                        default='configs/test.yaml',
                        help='Path to the YAML configuration file.')
    parser.add_argument('--model', type=str,
                        default='checkpoints/dacon_v1_1.pth',
                        help='Path to the DACoN weights.')
    parser.add_argument('--version', type=str,
                        default=None,
                        help='version of DACoN architecture.')
    args = parser.parse_args()

    main(args)