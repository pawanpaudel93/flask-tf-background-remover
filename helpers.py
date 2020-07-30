import cv2
import numpy as np
import os
import PIL.Image
import sys

from model import get_model

# COCO dataset object names
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

def apply_mask(image, mask):
    image[:, :, 0] = np.where(
        mask == 0,
        125,
        image[:, :, 0]
    )
    image[:, :, 1] = np.where(
        mask == 0,
        12,
        image[:, :, 1]
    )
    image[:, :, 2] = np.where(
        mask == 0,
        15,
        image[:, :, 2]
    )
    return image

# This function is used to show the object detection result in original image.
def display_instances(image, boxes, masks, ids, names, scores):
    # max_area will save the largest object for all the detection results
    max_area = 0
    
    # n_instances saves the amount of all objects
    n_instances = boxes.shape[0]

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue

        # compute the square of each object
        y1, x1, y2, x2 = boxes[i]
        square = (y2 - y1) * (x2 - x1)

        # use label to select person object from all the 80 classes in COCO dataset
        label = names[ids[i]]
        if label == 'person':
            # save the largest object in the image as main character
            # other people will be regarded as background
            if square > max_area:
                max_area = square
                mask = masks[:, :, i]
            else:
                continue
        else:
            continue

        # apply mask for the image
    # by mistake you put apply_mask inside for loop or you can write continue in if also
    image = apply_mask(image, mask)
        
    return image

def white_background(image):
    image = image.convert('RGBA')
    L,H = image.size
    color_0 = image.getpixel((0, 0))
    for h in range(H):
        for l in range(L):
            dot = (l, h)
            color_1 = image.getpixel(dot)
            if color_1 == color_0:
                color_1 = (255,255,255)
                image.putpixel(dot, color_1)
    return image

def transparent_background(image):
    image = image.convert('RGBA')
    L,H = image.size
    color_0 = image.getpixel((0, 0))
    for h in range(H):
        for l in range(L):
            dot = (l, h)
            color_1 = image.getpixel(dot)
            if color_1 == color_0:
                color_1 = color_1[:-1] + (0,)
                image.putpixel(dot, color_1)
    return image

def remove_bg(model, path, output_filepath, graph):
    image = np.array(PIL.Image.open(path))
    with graph.as_default():
        results = model.detect([image], verbose=0)
        r = results[0]
    image = display_instances(
        image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
    )
    image = PIL.Image.fromarray(np.array(image, dtype=np.uint8))
    image = transparent_background(image)
    output_filepath = output_filepath.split(".")
    output_filepath[-1] = "png"
    output_filepath = ".".join(output_filepath)
    image.save(output_filepath)
    return output_filepath.split('/')[-1]