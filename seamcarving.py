"""
Seam Carving Implementation for CV course @WWU
"""
import argparse
import sys
import os
import cv2
import numpy as np

def remove_vertical_seam(image, image_energy):
    """
    Removes low-energy vertical seams from an image based on the energy of individual pixels
    """
    #TODO: whole function
    image_result = image
    image_energy_result = image_energy
    return (image_result, image_energy_result)

def derive(image, operator):
    """
    Calculates pixel energy gradients based on sobel operator
    """
    image_height, image_width = image.shape
    gradients_result = np.full((image_height, image_width), 255)
    for row in range(0, image_height - 2):
        for column in range(0, image_width - 2):
            #REMINDER: index starts at 0, range at 1, that's why 3 instead of 2
            gradients_result[row + 1, column + 1] = np.sum(
                operator * image[row:row + 3, column:column + 3])
    return gradients_result

def energy(image):
    """
    Calculates the energy of each pixel in an image
    Uses sobel operator and convolution to calculate gradients
    """
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]])
    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]])
    gradients_x = derive(image, sobel_x)
    gradients_y = derive(image, sobel_y)
    energy_result = np.abs(gradients_x) +  np.abs(gradients_y)
    return energy_result

def convert_to_grayscale(image_rgb):
    """
    Converts and RGB Image to Grayscale
    """
    red, green, blue = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]
    image_gray = 0.2989 * red + 0.5870 * green + 0.1140 * blue
    return image_gray

def resize_image(image, vertical_seams_to_remove, horizontal_seams_to_remove):
    """
    Resizes and Image by removing low-energy horizontal and vertical seams
    """
    image_grayscale = convert_to_grayscale(image)
    image_energy = energy(image_grayscale)
    for _ in range(vertical_seams_to_remove):
        image, image_energy = remove_vertical_seam(image, image_energy)
    #TODO: order of vertical and horizontal seams
    for _ in range(horizontal_seams_to_remove):
        image, image_energy = remove_vertical_seam(image, image_energy)
    return image

def main():
    """
    Main function to read and write the file
    """
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

if __name__ == "__main__":
    main()
