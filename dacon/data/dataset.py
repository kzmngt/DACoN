
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import get_image, normalize_color, normalize_coordinate, get_seg_info


class DACoNDataset(Dataset):
    def __init__(self, data_list, data_root, seg_size = None, mode = "train"):
        self.data_list = data_list
        self.data_root = data_root
        self.seg_size = seg_size
        self.mode = mode #train, val_cf, val_kf

    def __len__(self):
        return len(self.data_list)
    
    def load_frame_data(self, char_name, frame_idx, frame_type):
        frame_idx_str = str(frame_idx).zfill(4)

        image_path = os.path.join(self.data_root, char_name, "gt", f"{frame_idx_str}.png")
        line_image_path = os.path.join(self.data_root, char_name, "line", f"{frame_idx_str}.png")
        seg_path = os.path.join(self.data_root, char_name, "seg", f"{frame_idx_str}.png")
        json_color_path = os.path.join(self.data_root, char_name, "json_color", f"{frame_idx_str}.json")

        if self.mode == "val_cf":
            json_color_path = os.path.join(self.data_root, char_name, "seg", f"{frame_idx_str}.json")
        elif self.mode == "val_kf":
            if frame_type == "tgt":
                json_color_path = os.path.join(self.data_root, char_name, "seg", f"{frame_idx_str}.json")
            elif frame_type == "src":
                image_path = os.path.join(self.data_root, char_name, "ref", "gt", f"{frame_idx_str}.png")
                line_image_path = os.path.join(self.data_root, char_name, "ref", "line", f"{frame_idx_str}.png")
                seg_path = os.path.join(self.data_root, char_name, "ref", "seg", f"{frame_idx_str}.png")
                json_color_path = os.path.join(self.data_root, char_name, "ref", "seg", f"{frame_idx_str}.json")

        color_image = get_image(image_path)
        line_image = get_image(line_image_path)

        seg_num, seg_sizes, seg_colors, seg_coords, seg_image = get_seg_info(seg_path, json_color_path, self.seg_size)

        H, W = seg_image.shape
        seg_colors = normalize_color(seg_colors)
        seg_coords = normalize_coordinate(seg_coords, (H, W))

        return {
            "color_image": color_image,
            "line_image": line_image,
            "seg_num": seg_num,
            "seg_sizes": seg_sizes,
            "seg_colors": seg_colors,
            "seg_coords": seg_coords,
            "seg_image": seg_image,
        }


    def __getitem__(self, idx):
        char_name = self.data_list[idx][0]
        frame_indices_src = self.data_list[idx][1]
        frame_indices_tgt = self.data_list[idx][2]

        # source frames
        color_images_src, line_images_src, seg_nums_src = [], [], []
        seg_sizes_src, seg_colors_src, seg_coords_src, seg_images_src = [], [], [], []

        for frame_idx in frame_indices_src:
            frame_type = "src"
            data = self.load_frame_data(char_name, frame_idx, frame_type)
            color_images_src.append(data["color_image"])
            line_images_src.append(data["line_image"])
            seg_nums_src.append(data["seg_num"])
            seg_sizes_src.append(data["seg_sizes"])
            seg_colors_src.append(data["seg_colors"])
            seg_coords_src.append(data["seg_coords"])
            seg_images_src.append(data["seg_image"])


        # target frames
        images_gt, line_images_tgt, seg_nums_tgt = [], [], []
        seg_sizes_tgt, seg_colors_tgt, seg_coords_tgt, seg_images_tgt = [], [], [], []

        for frame_idx in frame_indices_tgt:
            frame_type = "tgt"
            data = self.load_frame_data(char_name, frame_idx, frame_type)
            images_gt.append(data["color_image"])
            line_images_tgt.append(data["line_image"])
            seg_nums_tgt.append(data["seg_num"])
            seg_sizes_tgt.append(data["seg_sizes"])
            seg_colors_tgt.append(data["seg_colors"])
            seg_coords_tgt.append(data["seg_coords"])
            seg_images_tgt.append(data["seg_image"])


        #padding at frame dimension
        seg_sizes_src = pad_sequence(seg_sizes_src, batch_first=True, padding_value=-1)
        seg_sizes_tgt = pad_sequence(seg_sizes_tgt, batch_first=True, padding_value=-1)

        seg_coords_src = pad_sequence(seg_coords_src, batch_first=True, padding_value=-1)
        seg_coords_tgt = pad_sequence(seg_coords_tgt, batch_first=True, padding_value=-1)

        seg_colors_src = pad_sequence(seg_colors_src, batch_first=True, padding_value=-1)
        seg_colors_tgt = pad_sequence(seg_colors_tgt, batch_first=True, padding_value=-1)

        return {
            "char_name": char_name,
            "frame_indices_src": frame_indices_src,
            "frame_indices_tgt": frame_indices_tgt,
            "color_images_src": color_images_src,
            "line_images_src": line_images_src,
            "line_images_tgt": line_images_tgt,
            "images_gt": images_gt,
            "seg_nums_src": seg_nums_src,
            "seg_nums_tgt": seg_nums_tgt,
            "seg_sizes_src": seg_sizes_src,
            "seg_sizes_tgt": seg_sizes_tgt,
            "seg_colors_src": seg_colors_src,
            "seg_colors_tgt": seg_colors_tgt,
            "seg_coords_src": seg_coords_src,
            "seg_coords_tgt": seg_coords_tgt,
            "seg_images_src": seg_images_src,
            "seg_images_tgt": seg_images_tgt,
        }
 

class DACoNSingleDataset(Dataset):
    
    def __init__(self, data_list, data_root, is_ref = True, mode = "val_kf"):
        self.data_list = data_list
        self.data_root = data_root
        self.seg_size = None
        self.is_ref = is_ref
        self.mode = mode

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        char_name = self.data_list[idx][0]
        frame_name = self.data_list[idx][1]
        if self.mode == "val_kf":
            frame_name = str(frame_name).zfill(4)
            frame_name_ext = f"{frame_name}.png"
        else:
            frame_name_ext = frame_name
            frame_name = os.path.splitext(frame_name)[0]
        
        if self.is_ref:
            color_image_path = os.path.join(self.data_root, char_name, "ref", "gt", frame_name_ext)
            line_image_path = os.path.join(self.data_root, char_name, "ref", "line", frame_name_ext)
            seg_path = os.path.join(self.data_root, char_name, "ref", "seg", f"{frame_name}.png")
            json_color_path = os.path.join(self.data_root, char_name, "ref", "seg", f"{frame_name}.json")
        else:
            color_image_path = os.path.join(self.data_root, char_name, "gt", frame_name_ext)
            line_image_path = os.path.join(self.data_root, char_name, "line", frame_name_ext)
            seg_path = os.path.join(self.data_root, char_name, "seg", f"{frame_name}.png")
            json_color_path = os.path.join(self.data_root, char_name, "seg", f"{frame_name}.json")
        
        if self.mode == "infer" and not(self.is_ref):
            color_image = torch.zeros((0,), dtype=torch.float32)
            json_color_path = None
        else:
            color_image = get_image(color_image_path)

        line_image = get_image(line_image_path)
        
        
        seg_num, seg_sizes, seg_colors, seg_coords, seg_image = get_seg_info(seg_path, json_color_path, self.seg_size)
      
        H, W = seg_image.shape
        seg_colors = normalize_color(seg_colors)
        seg_coords = normalize_coordinate(seg_coords, (H, W))

        return {
            "char_name": char_name,
            "frame_name": frame_name,
            "color_image": color_image,
            "line_image": line_image,
            "seg_num": seg_num,
            "seg_sizes": seg_sizes,
            "seg_colors": seg_colors,
            "seg_coords": seg_coords,
            "seg_image": seg_image,
        }
