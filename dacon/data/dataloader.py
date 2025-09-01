import torch
from torch.nn.utils.rnn import pad_sequence

def dacon_pad_collate_fn(batch):
    char_names = [item['char_name'] for item in batch]
    frame_indices_src = [item['frame_indices_src'] for item in batch]
    frame_indices_tgt = [item['frame_indices_tgt'] for item in batch]

    def stack_nested_images(key):
        return [torch.stack(item[key]) for item in batch]  
    
    def process_segment_data(data_key: str, padding_value: float):

        transposed_data_list = [item[data_key].transpose(0, 1) for item in batch]
        padded_tensor = pad_sequence(transposed_data_list, batch_first=True, padding_value=padding_value)

        return padded_tensor.transpose(1, 2).contiguous()
    
    color_images_src = torch.stack(stack_nested_images('color_images_src'))  # [B, N, C, H, W]
    line_images_src = torch.stack(stack_nested_images('line_images_src'))
    line_images_tgt = torch.stack(stack_nested_images('line_images_tgt'))
    images_gt = torch.stack(stack_nested_images('images_gt'))
    seg_images_src = torch.stack(stack_nested_images('seg_images_src'))
    seg_images_tgt = torch.stack(stack_nested_images('seg_images_tgt'))

    seg_nums_src = torch.tensor([item['seg_nums_src'] for item in batch])
    seg_nums_tgt = torch.tensor([item['seg_nums_tgt'] for item in batch])

    seg_sizes_src = process_segment_data('seg_sizes_src', -1.0)
    seg_sizes_tgt = process_segment_data('seg_sizes_tgt', -1.0)
    seg_colors_src = process_segment_data('seg_colors_src', -1.0)
    seg_colors_tgt = process_segment_data('seg_colors_tgt', -1.0)
    seg_coords_src = process_segment_data('seg_coords_src', -1.0)
    seg_coords_tgt = process_segment_data('seg_coords_tgt', -1.0)

    return {
        "char_name": char_names,
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

def dacon_single_pad_collate_fn(batch):
    
    char_name = [item['char_name'] for item in batch]
    frame_name = [item['frame_name'] for item in batch]
    color_image = torch.stack([item['color_image'] for item in batch])
    line_image = torch.stack([item['line_image'] for item in batch])
    seg_num = torch.tensor([item['seg_num'] for item in batch])
    seg_sizes = [item['seg_sizes'] for item in batch]
    seg_colors = [item['seg_colors'] for item in batch]
    seg_coords = [item['seg_coords'] for item in batch]
    seg_image = torch.stack([item['seg_image'] for item in batch])

    padded_sizes = pad_sequence(seg_sizes, batch_first=True, padding_value=-1.0)
    padded_colors = pad_sequence(seg_colors, batch_first=True, padding_value=-1.0)
    padded_coords = pad_sequence(seg_coords, batch_first=True, padding_value=-1.0)

    return {
        "char_name": char_name,
        "frame_name": frame_name,
        "color_image": color_image,
        "line_image": line_image,
        "seg_num": seg_num,
        "seg_sizes": padded_sizes,
        "seg_colors": padded_colors,
        "seg_coords": padded_coords,
        "seg_image": seg_image,
    }

