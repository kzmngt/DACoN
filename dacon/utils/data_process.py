import os
import json
import glob
import torch
import random
from torchvision.utils import save_image

def save_image_pred(image_pred, char_name, frame_name, save_path):

    if isinstance(frame_name, int):
        frame_name = str(frame_name).zfill(4)

    folder_path = os.path.join(save_path, "images", char_name)
    os.makedirs(folder_path, exist_ok=True)
    file_path  = os.path.join(folder_path, f"{frame_name}.png")

    image_pred = image_pred.permute(2, 0, 1)

    save_image(image_pred, file_path)
    
    return

def save_json_pred(color_list_pred, char_name, frame_name, save_path):
    
    if isinstance(frame_name, int):
        frame_name = str(frame_name).zfill(4)
    
    folder_path = os.path.join(save_path, "json", char_name)
    os.makedirs(folder_path, exist_ok=True)
    json_file_path  = os.path.join(folder_path, f"{frame_name}.json")
    
    color_dict = {str(idx + 1): [int(value) for value in color.tolist()] for idx, color in enumerate(color_list_pred)}
    
    with open(json_file_path, "w") as json_file:
        json.dump(color_dict, json_file)

    return

def find_latest_checkpoint(model_save_path):
    checkpoint_files = glob.glob(os.path.join(model_save_path, "model_epoch_*.pt"))
    if not checkpoint_files:
        return None
    
    checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return checkpoint_files[-1] 

def normalize_color(color):

    return color / 255

def normalize_coordinate(coords, seg_size):
    
    H, W = seg_size

    norm_factors = torch.tensor([H, W, H, W], 
                              dtype=torch.float32)
    coordinate_tensor = coords / norm_factors

    return coordinate_tensor

def normalize_coordinate_center(coords, image):

    _, H, W = image.shape
    device = coords.device
    center = torch.tensor([W / 2, H / 2, W / 2, H / 2], dtype=torch.float32, device=device)
    scale  = torch.tensor([W / 2, H / 2, W / 2, H / 2], dtype=torch.float32, device=device)

    coords_norm = (coords - center) / scale
    return coords_norm

def normalize_size(sizes, image):
    
    _, image_height, image_width = image.shape
    image_size = image_height*image_width

    return sizes / image_size

def move_data_to_device(data, device):
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device).to(torch.float32)
    return data

def get_folder_names(path):
    folder_names = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    return folder_names

def get_file_names(path):
    file_names = [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
    return file_names

def get_file_count(path):
    file_names = [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
    return len(file_names)

def make_train_data_list(data_root):
    
    data_list = []
    char_names = get_folder_names(data_root)
    
    for char_name in char_names:
        frame_num = get_file_count(os.path.join(data_root, char_name, "line"))
        for i in range(frame_num):
                
            value = random.choice([0, 1, 2])
            sign = random.choice([-1, 1])
            src_idx = i + value * sign
            
            if src_idx >= frame_num:
                src_idx %= frame_num
            elif src_idx < 0:
                src_idx = frame_num + src_idx

            data_list.append([char_name, [src_idx], [i]]) #char_name, frame_idx_src, frame_idx_tgt
    
    return data_list

def make_val_data_list(data_root, colorize_type, clip_interval):
    
    data_list = []
    char_names = get_folder_names(data_root)
    max_clip = False
    if clip_interval == "max":
        max_clip = True
    
    for char_name in char_names:
        frame_num = get_file_count(os.path.join(data_root, char_name, "line"))

        if max_clip:
            clip_interval = frame_num
        
        for i in range(frame_num):

            if colorize_type == "keyframe":
                data_list.append([char_name, [0], [i]]) #char_name, frame_idx_src, frame_idx_tgt
                    
            elif colorize_type == "consecutive_frame":
                if (i+1)%clip_interval != 0:
                    data_list.append([char_name, [i], [i+1]]) #char_name, frame_idx_src, frame_idx_tgt

    return data_list

def make_single_data_list(data_root, char_name, ref_shot, is_ref):
    
    data_list = []

    if is_ref:
        if ref_shot == "max":
            frame_num = get_file_count(os.path.join(data_root, char_name, "ref", "gt"))
        else:
            frame_num =ref_shot
    else:
        frame_num = get_file_count(os.path.join(data_root, char_name, "line"))
    
    for i in range(frame_num):

        data_list.append([char_name, i]) #char_name, frame_idx


    return data_list

def make_inference_data_list(data_root, char_name, is_ref):
    
    data_list = []
    
    if is_ref:
        frame_names = get_file_names(os.path.join(data_root, char_name, "ref", "gt"))
    else:
        frame_names = get_file_names(os.path.join(data_root, char_name, "line"))

    for frame_name in frame_names:
        data_list.append([char_name, frame_name])

    return data_list




