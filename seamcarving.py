"""
Seam Carving Implementation for CV course @WWU
"""
import argparse
import sys
import os
import cv2
import numpy as np

def find_vertical_seam(image_energy):
    """
    Finds a vertical seam by calculating the cumulative energy
    """
    image_height, image_width = image_energy.shape
    #Reminder that python will never copy objects
    cumulative_min_energy = image_energy.copy()
    #iterate through through pixel by column then by row
    for row in range(1, image_height):
        for column in range(image_width):
            #cover first and last column edge-cases
            if column == 0:
                #don't consider previous column
                cumulative_min_energy[row, column] = image_energy[row, column] + min(
                    cumulative_min_energy[row - 1, column:column + 1])
            elif column == image_width - 1:
                #don't consider next column
                cumulative_min_energy[row, column] = image_energy[row, column] + min(
                    cumulative_min_energy[row - 1, column - 1:column])
            else:
                #calculate minimal energy required to reach current pixel from previous row
                cumulative_min_energy[row, column] = image_energy[row, column] + min(
                    cumulative_min_energy[row - 1, column - 1:column + 1])
    return cumulative_min_energy

def remove_vertical_seam(image, image_energy):
    """
    Finds and removes lowest-energy vertical seam from an image
    """
    image_height, image_width = image_energy.shape
    #create empty resulting image and energy image
    image_result = np.zeros((image_height, image_width - 1, 3))
    image_energy_result = np.zeros((image_height, image_width - 1))
    #calculate energy of seams
    cumulative_min_energy = find_vertical_seam(image_energy)
    #find seam with lowest energy
    seam_column = np.argmin(cumulative_min_energy[image_height - 1, :])
    #reverse iterate through rows to remove pixel of lowest-energy seam
    for row in reversed(range(image_height - 1)):
        #delete pixel of lowest-energy seam
        image_result[row, :] = np.delete(image[row, :], seam_column, 0)
        #delete pixel from energy image to ensure consitency
        image_energy_result[row, :] = np.delete(image_energy[row, :], seam_column, 0)
        #cover first and last column edge-cases
        if seam_column == 0:
            seam_column = seam_column + np.argmin(
                cumulative_min_energy[row - 1, seam_column:seam_column + 1])
        elif seam_column == image_width - 1:
            seam_column = seam_column + np.argmin(
                cumulative_min_energy[row - 1, seam_column - 1:seam_column]) - 1
        else:
            #find pixel of lowest-energy seam in above row
            seam_column = seam_column + np.argmin(
                cumulative_min_energy[row - 1, seam_column - 1:seam_column + 1]) - 1
    return (image_result, image_energy_result)

def find_horizontal_seam(image_energy):
    """
    Finds a horizontal seam by calculating the cumulative energy
    -> see vertical version for documentation
    """
    image_height, image_width = image_energy.shape
    cumulative_min_energy = image_energy.copy()
    for column in range(1, image_width):
        for row in range(image_height):
            if row == 0:
                cumulative_min_energy[row, column] = image_energy[row, column] + min(
                    cumulative_min_energy[row:row + 1, column - 1])
            elif row == image_height - 1:
                cumulative_min_energy[row, column] = image_energy[row, column] + min(
                    cumulative_min_energy[row - 1:row, column - 1])
            else:
                cumulative_min_energy[row, column] = image_energy[row, column] + min(
                    cumulative_min_energy[row - 1:row + 1, column - 1])
    return cumulative_min_energy

def remove_horizontal_seam(image, image_energy):
    """
    Finds and removes lowest-energy horizontal seam from an image
    -> see vertical version for documentation
    """
    image_height, image_width = image_energy.shape
    image_result = np.zeros((image_height - 1, image_width, 3))
    image_energy_result = np.zeros((image_height - 1, image_width))
    cumulative_min_energy = find_horizontal_seam(image_energy)
    seam_row = np.argmin(cumulative_min_energy[:, image_width - 1])
    for column in reversed(range(image_width - 1)):
        image_result[:, column] = np.delete(image[:, column], seam_row, 0)
        image_energy_result[:, column] = np.delete(image_energy[:, column], seam_row, 0)
        if seam_row == 0:
            seam_row = seam_row + np.argmin(
                cumulative_min_energy[seam_row:seam_row + 1, column - 1])
        elif seam_row == image_height - 1:
            seam_row = seam_row + np.argmin(
                cumulative_min_energy[seam_row - 1:seam_row, column - 1]) - 1
        else:
            seam_row = seam_row + np.argmin(
                cumulative_min_energy[seam_row - 1:seam_row + 1, column - 1]) - 1
    return (image_result, image_energy_result)

def derive(image, operator):
    """
    Calculates pixel energy gradients based on sobel operator
    """
    image_height, image_width = image.shape
    #set starting value to max since we will be searching for minima
    gradients_result = np.full((image_height, image_width), 255)
    #iterate through pixels
    for row in range(0, image_height - 2):
        for column in range(0, image_width - 2):
            #calculate gradient for current pixel
            #REMINDER TO NOT FUCK THIS UP: index starts at 0, range at 1
            gradients_result[row + 1, column + 1] = np.sum(
                operator * image[row:row + 3, column:column + 3])
    return gradients_result

def energy(image):
    """
    Calculates the energy of each pixel in an image
    Uses sobel operator and convolution to calculate gradients
    See equation (1) from Avidan et al.
    """
    #create sobel operators for gradient calculation
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]])
    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]])
    #calculate gradients
    gradients_x = derive(image, sobel_x)
    gradients_y = derive(image, sobel_y)
    #calculate energy as defined in Avidan et al.
    energy_result = np.abs(gradients_x) +  np.abs(gradients_y)
    return energy_result

def convert_to_grayscale(image_rgb):
    """
    Converts and RGB Image to Grayscale
    See Matlab implementation of rgb2gray for details
    """
    red, green, blue = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]
    image_gray = 0.2989 * red + 0.5870 * green + 0.1140 * blue
    return image_gray

def resize_image(image, vertical_seams_to_remove, horizontal_seams_to_remove):
    """
    Resizes and Image by removing low-energy horizontal and vertical seams
    """
    #convert image to grayscale
    print("-converting into grayscale...")
    image_grayscale = convert_to_grayscale(image)
    #calculate image energy
    print("-calculating energy...")
    image_energy = energy(image_grayscale)
    #remove given number of vertical seams
    for i in range(vertical_seams_to_remove):
        print("-removing vertical seam " + str(i+1) + "...")
        image, image_energy = remove_vertical_seam(image, image_energy)
    #remove given number of horizontal seams
    for j in range(horizontal_seams_to_remove):
        print("-removing horizontal seam " + str(j+1) + "...")
        image, image_energy = remove_horizontal_seam(image, image_energy)
    return image

def main():
    """
    Main function to read and write the file
    """
    #parse startup arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-file')
    parser.add_argument('-vertical', type=int)
    parser.add_argument('-horizontal', type=int)
    arguments = parser.parse_args()

    #save startup arguments
    image_file = arguments.file
    vertical_seams_to_remove = arguments.vertical
    horizontal_seams_to_remove = arguments.horizontal

    #check if both number of horizontal and vertical seams to remove have been entered
    if arguments.vertical is None and arguments.horizontal is None:
        sys.exit(0)
    elif arguments.vertical is None:
        vertical_seams_to_remove = 0
    elif arguments.horizontal is None:
        horizontal_seams_to_remove = 0

    #check if the given file exists
    if os.path.exists(image_file):
        #check if the given file is a supported image
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            #load image using OpenCV and save dimensions of original and new image
            image = cv2.imread(image_file)
            height, width, _ = image.shape
            new_height = int(height - horizontal_seams_to_remove)
            new_width = int(width - vertical_seams_to_remove)
            #check if image is big enough to remove given number of seams
            if new_width > 0 and new_height > 0:
                print("Starting to resize!")
                result = resize_image(image, vertical_seams_to_remove, horizontal_seams_to_remove)
                #save resized image with new name using OpenCV
                image_name = os.path.splitext(image_file)[0]
                image_output = image_name + '_' + str(new_width) + 'x' + str(new_height) + '.png'
                cv2.imwrite(image_output, result)
                print("Successfully exported!")
            else:
                print("You are trying to remove too many seams for this picture.")
                sys.exit(0)
        else:
            print("Incorrect fily type.")
            sys.exit(0)
    else:
        print("File not found.")
        sys.exit(0)

if __name__ == "__main__":
    main()
