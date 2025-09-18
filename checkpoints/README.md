# Quantitative Results

This README provides detailed quantitative results for the released DACoN models.  
Note that the results reported here differ slightly from those in the original paper.


|   Model    |   Download  |                                  Description                                                    |
| :--------: | :---------: | :---------------------------------------------------------------------------------------------: |
| DACoN v1.0 | [link](https://drive.google.com/file/d/1VvgLFwas_LcawrWh274BEpw2P_euOg3a/view?usp=sharing) |                                 Same architecture as paper                                      |
| DACoN v1.1 | [link](https://drive.google.com/file/d/1KJ77-aFDePmsJ6LDicJgM4pyGjagu6aI/view?usp=sharing) | Set a fixed image size during segment pooling.<br>Fuse DINO and U-Net features using simple addition instead of concat + MLP. |


## Table 1. Keyframe Colorization

| Model | Ref. shot |  Acc  | Acc-Thresh | Pix-Acc | Pix-F-Acc | Pix-B-MIoU |
|:-----:|:---------:|:-----:|-----------:|:-------:|:---------:|:----------:|
| v1.0  |     1     | 67.61 |     72.52  |  96.85  |   90.65   |    99.18   |
|       |     5     | 73.35 |     77.57  |  97.76  |   93.65   |    99.18   |
|       |    max    | 74.47 |     78.53  |  98.06  |   94.29   |    99.23   |
| v1.1  |     1     | 68.01 |     72.87  |  96.97  |   91.03   |    99.11   |
|       |     5     | 73.91 |     78.23  |  97.84  |   94.28   |    98.92   |
|       |    max    | 75.05 |     79.23  |  98.19  |   94.79   |    99.16   |

## Table 2. Consecutive Frame Colorization

| Model |  Input Data  |  Acc  | Acc-Thresh | Pix-Acc | Pix-F-Acc | Pix-B-MIoU |
|:-----:|:------------:|:-----:|:----------:|:-------:|:---------:|:----------:|
| v1.0  | 3D Rendered  | 84.61 |   88.14    |  99.27  |   97.97   |   99.63    |
|       | Hand-drawn   | 87.40 |   90.54    |  99.10  |   96.95   |   99.71    |
| v1.1  | 3D Rendered  | 85.32 |   88.82    |  99.34  |   98.24   |   99.65    |
|       | Hand-drawn   | 87.66 |   90.84    |  99.27  |   97.27   |   99.75    | 

---
> **Note:**  
> - "Ref. shot" indicates the number of reference images; "max" uses all images in the color design sheet for each character.  
> - Acc-Thresh is computed only for segments larger than 10 pixels.
