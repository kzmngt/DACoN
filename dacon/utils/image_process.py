import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.io import read_image
from skimage import measure
from skimage.morphology import footprint_rectangle, binary_dilation

def get_image(image_path):
    
    image_tensor = read_image(image_path).float()
    image_tensor /= 255.0
    
    return image_tensor

def get_alpha_line_image(image_path):

    image_tensor = read_image(image_path).float() / 255.0
    line_image_tensor = image_tensor[3:4, :, :]

    return line_image_tensor

def get_seg_idx_image(seg_image_path):
    
    seg_image = read_image(seg_image_path)
    seg_idx_image = (seg_image[0] << 16) + (seg_image[1] << 8) + seg_image[2]
    seg_list = torch.unique(seg_idx_image[seg_idx_image != 0])

    return seg_idx_image, seg_list


def get_seg_info(seg_image_path, json_colors_path, seg_size):
    
    seg_image = read_image(seg_image_path)

    if not(seg_size == None):
        seg_image = F.interpolate(seg_image.unsqueeze(0), size=seg_size, mode='nearest').squeeze(0)

    if not(json_colors_path == None):
        with open(json_colors_path, 'r') as file:
            color_data = json.load(file)
    
    _, H, W = seg_image.shape

    seg_idx_image = (seg_image[0] << 16) + (seg_image[1] << 8) + seg_image[2]
    seg_list = torch.unique(seg_idx_image[seg_idx_image != 0])
    
    seg_num = len(seg_list)
    seg_colors = torch.empty((seg_num, 4), dtype=torch.float32)
    seg_coordinates = torch.empty((seg_num, 4), dtype=torch.int64) 
    seg_sizes = torch.empty((seg_num), dtype=torch.float32) 
    
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

    new_seg_image = torch.zeros_like(seg_idx_image, device=seg_idx_image.device) 

    for idx, seg_idx in enumerate(seg_list):
        mask = seg_idx_image == seg_idx

        if not(json_colors_path == None):
            rgba_value = color_data.get(str(seg_idx.item()), [-1, -1, -1, -1])
            seg_colors[idx] = torch.tensor(rgba_value, dtype=torch.float32)

        seg_sizes[idx] = mask.sum()
        
        xs = xx[mask]
        ys = yy[mask]
        x_min = xs.min()
        x_max = xs.max()
        y_min = ys.min()
        y_max = ys.max()
        
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        mask_coordinate = [center_y, center_x, height, width]
        seg_coordinates[idx] = torch.tensor(mask_coordinate, dtype=torch.int64)

        new_seg_image = torch.where(mask, idx+1, new_seg_image)
        
    return seg_num, seg_sizes, seg_colors, seg_coordinates, new_seg_image


def save_seg_label(label_image, save_path):

    if save_path is None:
        return

    h, w = label_image.shape
    output_img = np.zeros((h, w, 3), dtype=np.uint8)

    label_uint = label_image.astype(np.int32)
    output_img[:, :, 0] = (label_uint >> 16) & 255
    output_img[:, :, 1] = (label_uint >> 8) & 255
    output_img[:, :, 2] = label_uint & 255

    Image.fromarray(output_img).save(save_path)

def merge_color_line_to_region(label_image, color_line_label_image):

    updated_label_image = label_image.copy()
    current_max_label = updated_label_image.max()

    color_labels = np.unique(color_line_label_image)
    color_labels = color_labels[color_labels != 0]

    for lbl in color_labels:
        line_mask = color_line_label_image == lbl
        dilated_mask = binary_dilation(line_mask, footprint_rectangle((3, 3)))

        surrounding_labels = updated_label_image[dilated_mask]
        surrounding_labels = surrounding_labels[surrounding_labels != 0]

        if len(surrounding_labels) > 0:
            target_label = np.bincount(surrounding_labels).argmax()
        else:
            current_max_label += 1
            target_label = current_max_label

        updated_label_image[line_mask] = target_label

    return updated_label_image

def label_color_regions(filtered_image):
    label_img = np.zeros(filtered_image.shape[:2], dtype=np.int32)
    current_label = 1

    mask_alpha = filtered_image[:, :, 3] > 0
    colored_pixels = filtered_image[mask_alpha][:, :3]

    unique_colors = np.unique(colored_pixels, axis=0)

    for color in unique_colors:
        color_mask = np.all(filtered_image[:, :, :3] == color, axis=-1) & mask_alpha
        labeled = measure.label(color_mask, connectivity=2)

        labeled_nonzero = labeled > 0
        label_img[labeled_nonzero] = labeled[labeled_nonzero] + current_label - 1

        current_label += labeled.max()

    return label_img

def extract_color_line(line_rgba):
    mask = (line_rgba[:, :, 3] > 0) & (~np.all(line_rgba[:, :, :3] == [0, 0, 0], axis=2))

    filtered_image = np.ones_like(line_rgba, dtype=np.uint8) * 255
    filtered_image[:, :, 3] = 0

    filtered_image[mask] = line_rgba[mask]
    filtered_image[mask, 3] = 255

    label_image = np.zeros(filtered_image.shape[:2], dtype=np.int32)
    current_label = 1

    mask_alpha = filtered_image[:, :, 3] > 0
    colored_pixels = filtered_image[mask_alpha][:, :3]

    unique_colors = np.unique(colored_pixels, axis=0)

    for color in unique_colors:
        color_mask = np.all(filtered_image[:, :, :3] == color, axis=-1) & mask_alpha
        labeled = measure.label(color_mask, connectivity=2)
        labeled_nonzero = labeled > 0
        label_image[labeled_nonzero] = labeled[labeled_nonzero] + current_label - 1

        current_label += labeled.max()

    return label_image

def label_closed_regions(line_rgba):
    line_mask = line_rgba[:, :, 3] > 0

    region_mask = ~line_mask
    label_image = measure.label(region_mask, connectivity=1) 

    return label_image

def convert_to_line_rgba(image_path):
    image = Image.open(image_path)
    image_np = np.array(image)

    h, w = image_np.shape[:2]
    line_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    if image.mode == "RGBA":
        line_rgba[:, :, :3] = image_np[:, :, :3]
        line_rgba[:, :, 3] = np.where(image_np[:, :, 3] > 0, 255, 0)
    
    elif image.mode == "RGB":
        line_rgba[:, :, :3] = image_np
        mask = np.any(image_np[:, :, :3] < 255, axis=2)
        line_rgba[:, :, 3] = np.where(mask, 255, 0)

    return line_rgba

def extract_segment(line_image_path, save_path):
    line_rgba = convert_to_line_rgba(line_image_path)
    color_line_label_image = extract_color_line(line_rgba)
    label_image = label_closed_regions(line_rgba)
    label_image = merge_color_line_to_region(label_image, color_line_label_image)
    save_seg_label(label_image, save_path)

    return

def extract_color(color_image_path, seg_image_path, save_path):
    color_image = read_image(color_image_path)  # torch.Tensor [C,H,W]
    
    if isinstance(color_image, torch.Tensor):
        color_image = color_image.permute(1, 2, 0).cpu().numpy()  # [H,W,C]

    if color_image.shape[2] == 3:
        alpha = np.full((color_image.shape[0], color_image.shape[1], 1), 255, dtype=np.uint8)
        color_image = np.concatenate([color_image, alpha], axis=-1)
    
    seg_idx_image, _ = get_seg_idx_image(seg_image_path)
    if hasattr(seg_idx_image, "cpu"):
        seg_idx_image = seg_idx_image.cpu().numpy().astype(np.int32)

    color_dict = {}
    props = measure.regionprops(seg_idx_image)

    for i in range(1, seg_idx_image.max() + 1):
        coords = props[i - 1].coords
        region_colors = color_image[coords[:,0], coords[:,1], :]
        
        unique_colors, counts = np.unique(region_colors, axis=0, return_counts=True)
        most_common_color = unique_colors[np.argmax(counts)]
        color_dict[str(i)] = most_common_color.tolist()

    with open(save_path, "w") as f:
        json.dump(color_dict, f, indent=2)

def colorize_target_image(color_list_pred, image_tgt, seg_image_tgt):

    image_tgt = image_tgt.permute(1,2,0)
    combined_image = torch.zeros_like(image_tgt, device=image_tgt.device) 

    for i, color in enumerate(color_list_pred):
        mask = seg_image_tgt == i+1
        mask = mask.unsqueeze(-1).expand(-1, -1, 4).bool()
        combined_image = torch.where(mask, color/255, combined_image)
    
    black_mask = (image_tgt == torch.tensor([0, 0, 0, 1], device=image_tgt.device)).all(dim=-1)
    combined_image[black_mask] = image_tgt[black_mask]

    return combined_image

