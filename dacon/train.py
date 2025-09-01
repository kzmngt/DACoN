import os
import sys
import time
import yaml
import argparse
import datetime
import torch
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader

from test import validate
from models import DACoNModel
from data import DACoNDataset, dacon_pad_collate_fn
from losses import CombinedLoss
from utils import (
    make_train_data_list,
    make_val_data_list,
    move_data_to_device,
    load_config,
    format_time,
    find_latest_checkpoint,
    setup_logger,
)



def main(args):
    config = load_config(args.config)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Hyperparameters from config ---
    version = args.version
    if version == None:
        version = config['version']
    colorize_type = config['colorize_type']

    train_data_root = config['datasets']['train']['root']
    val_data_root = config['datasets']['val']['root']
    num_workers_train = config['datasets']['train']['num_worker']
    num_workers_val = config['datasets']['val']['num_worker']
    clip_interval = config['datasets']['clip_interval']

    num_epochs = config['train']['num_epochs']
    train_batch_size = config['train']['batch_size']
    learning_rate = float(config['train']['scheduler']['learning_rate'])
    model_save_path = config['train']['model_save_path']
    os.makedirs(model_save_path, exist_ok=True)

    loss_scale_ce = config['losses']['loss_scale_ce']
    loss_scale_mae = config['losses']['loss_scale_mae']

    val_batch_size = config['val']['batch_size']
    save_images = config['val']['save_images']
    save_json = config['val']['save_json']
    save_path = config['val']['save_path']
    if colorize_type == "keyframe":
        save_path = os.path.join(save_path, "train", f"{colorize_type}_1")
    elif colorize_type == "consecutive_frame":
        save_path = os.path.join(save_path, "train", colorize_type)
    os.makedirs(save_path, exist_ok=True)

    seed_num = config['manual_seed']
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)

    logger = setup_logger(save_path, current_time, log_name="dacon_train")
    logger.info(f"Training DACoN version {version}.")
    logger.info("\n===== Config =====\n" + yaml.dump(config, default_flow_style=False, sort_keys=False))

    device = torch.device("cuda" if torch.cuda.is_available() and config['num_gpu'] > 0 else "cpu")
    logger.info(f"Using device: {device}")

    model = DACoNModel(config['network'], version).to(device)
    logger.info("\n===== Model Architecture =====\n" + str(model))

    val_data_list = make_val_data_list(val_data_root, colorize_type, clip_interval)
    if colorize_type == "keyframe":
        val_dataset = DACoNDataset(val_data_list, val_data_root, mode = "val_kf")
    elif colorize_type == "consecutive_frame":
        val_dataset = DACoNDataset(val_data_list, val_data_root, mode = "val_cf")
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers_val, collate_fn=dacon_pad_collate_fn)

    criterion = CombinedLoss(loss_scale_ce=loss_scale_ce, loss_scale_mae=loss_scale_mae).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    logger.info("Loss function and Optimizer initialized.")

        
    latest_ckpt = find_latest_checkpoint(model_save_path)
    start_epoch = 0
    losses = []

    if latest_ckpt is not None:
        logger.info(f"Loading checkpoint {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model_state = model.state_dict()
        filtered_state = checkpoint['model_state_dict']
        model_state.update(filtered_state)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        losses = checkpoint['losses']
        start_epoch = checkpoint['epoch']

    total_training_start_time = time.time()
    logger.info("--- Start Training ---")


    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0

        train_data_list = make_train_data_list(train_data_root)
        train_dataset = DACoNDataset(train_data_list, train_data_root, seg_size = config['datasets']['train']['seg_size'],  mode = "train")
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=config['datasets']['train']['use_shuffle'], num_workers=num_workers_train, collate_fn=dacon_pad_collate_fn)

        num_train_batches = len(train_dataloader)
        logger.info(f"--- Epoch {epoch + 1}/{num_epochs} ---")
        logger.info(f"  Training on {len(train_dataset)} samples ({num_train_batches} batches) with batch size {train_batch_size}.")


        for i, data in enumerate(train_dataloader):
            data = move_data_to_device(data, device)

            seg_sim_map, dino_seg_sim_map  = model.forward(data)
            loss, ce_loss, mae_loss  = criterion(data, seg_sim_map, dino_seg_sim_map)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   

            running_loss += loss.item()
            print(f"  Batch {i + 1}/{num_train_batches} - Current Loss: {loss.item():.4f}", end='\r')

            del data, seg_sim_map, dino_seg_sim_map, loss, ce_loss, mae_loss
            torch.cuda.empty_cache()

            
        average_loss = running_loss / len(train_dataloader)
        losses.append(average_loss)
        logger.info(f"  Epoch {epoch + 1} Training Average Loss: {average_loss:.4f}")

        del average_loss, running_loss
        torch.cuda.empty_cache()

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_elapsed_time = epoch_end_time - total_training_start_time
        avg_epoch_duration = total_elapsed_time / (epoch + 1)
        estimated_remaining_time_seconds = avg_epoch_duration * (num_epochs - (epoch + 1))


        logger.info(f"  Epoch Duration: {format_time(epoch_duration)} | "
                    f"Total Elapsed: {format_time(total_elapsed_time)} | "
                    f"Estimated Remaining: {format_time(estimated_remaining_time_seconds)}")


        if (epoch + 1) % 1 == 0:
            model.eval()
            with torch.no_grad():
                logger.info(f"  Validating on {len(val_dataloader)} samples ({len(val_dataloader)} batches) with batch size {val_batch_size}.")
                val_results = validate(model, val_dataloader, criterion, device, save_images, save_json, save_path)
    
            logger.info(
                        f"Validation Results (Epoch {epoch + 1}):\n"
                        f"  Val Average Loss: {val_results['loss']:.4f}\n"
                        f"  Val Segment Accuracy: {val_results['seg_acc']:.4f} (Threshold: {val_results['seg_acc_thres']:.4f})\n"
                        f"  Val Pixel Accuracy: {val_results['pix_acc']:.4f}"
                        )

            del val_results
            torch.cuda.empty_cache()

            model_state = model.state_dict()
            filtered_state = {k: v for k, v in model_state.items()if not (k.startswith('dino.'))}

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': filtered_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses
            }
            model_save_filepath = f'{model_save_path}/model_epoch_{epoch+1}.pt'
            torch.save(checkpoint, model_save_filepath)
            logger.info(f"  Model checkpoint saved to {model_save_filepath}")

    total_finish_time = time.time()
    logger.info(f"\n--- Training complete! ---"
                f"\nTotal training time: {format_time(total_finish_time - total_training_start_time)}")


    final_model_state = model.state_dict()
    filtered_state = {k: v for k, v in final_model_state.items() if not (k.startswith('dino.'))}
    final_model_path = os.path.join(model_save_path, f"dacon_{current_time}.pth")

    torch.save(filtered_state, final_model_path) 
    # torch.save(final_model_state, final_model_path)  # if save all weights

    logger.info(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.append(str(ROOT))

    parser = argparse.ArgumentParser(description="Train the DACoN model.")
    parser.add_argument('--config', type=str,
                        default='configs/train.yaml',
                        help='Path to the YAML configuration file.')
    parser.add_argument('--version', type=str,
                        default=None,
                        help='version of DACoN architecture.')
    args = parser.parse_args()

    main(args)