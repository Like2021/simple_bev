import cv2
import numpy as np
from PIL import Image
import matplotlib
# matplotlib
import matplotlib.pyplot as plt
import os
# from nets.ops


def vis_attn_map(img_path, attn_map, ratio=0.5, cmap="jet"):
    """

    """
    print(f"loading image from {img_path}")
    # load the image
    img = Image.open(img_path, mode="r")
    img_H, img_W = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_H, 0.02 * img_W))

    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')

    # normalize the attention mask
    mask = cv2.resize(attn_map, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)


img_path = ".././Fast-BEV/data/nuscenes/samples/CAM_FRONT/" \
           "n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915243012465.jpg"
attn_map = np.random.random((2, 5))
vis_attn_map(img_path, attn_map)
