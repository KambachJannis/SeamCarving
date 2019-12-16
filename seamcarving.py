import argparse
import sys
import os
import cv2
import numpy as np

def resize_image(image, vertical_seams_to_remove, horizontal_seams_to_remove):
    image_grayscale = convert_to_grayscale(image)
    image_energy = energy(image_grayscale)
    return 0

# Add Documentation
parser = argparse.ArgumentParser()
parser.add_argument('-f')
parser.add_argument('-v')
parser.add_argument('-h')
arguments = parser.parse_args()

image_file = arguments.f
vertical_seams_to_remove = arguments.v
horizontal_seams_to_remove = arguments.h

if arguments.x is None and arguments.y is None:
    sys.exit(0)
elif arguments.x is None:
    vertical_seams_to_remove = 0
elif arguments.y is None:
    horizontal_seams_to_remove = 0

if os.path.exists(image_file):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imread(image_file)
        height, width, _ = image.shape
        new_height = int(height - horizontal_seams_to_remove)
        new_width = int(width - vertical_seams_to_remove)
        if new_width > 0 and new_height > 0:
            print('go for resizing')
            result = resize_image(image, vertical_seams_to_remove, horizontal_seams_to_remove)
            image_name = os.path.splitext(image_file)[0]
            image_name_output = image_name + '_' + new_width + 'x' + new_height + '.png'
            cv2.imwrite(image_name_output, result)
            print('exported')
        else:
            print('too many seams to remove')
            sys.exit(0)
    else:
        print('file type incorrect')
        sys.exit(0)
else:
    print('image not found')
    sys.exit(0)       