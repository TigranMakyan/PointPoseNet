import cv2
import numpy as np
import os

def calculate_mean_and_std(folder_path):
    '''
    Funtion to calculate mean and std for image folder. But my experiments show me that 
    without normalization the model can achieve better results
    '''
    # Initialize variables to accumulate the sum of pixel values
    total_pixels = 0
    sum_channel_values = np.zeros(3)

    # Get a list of image file names in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for i, image_file in enumerate(image_files):
        print(i)
        # Read the image using OpenCV
        image = cv2.imread(os.path.join(folder_path, image_file))
        
        # Convert the image to float32
        image = image.astype(np.float32) / 255.0

        # Calculate the sum of pixel values for each channel (R, G, B)
        sum_channel_values += np.sum(image, axis=(0, 1))
        
        # Update the total number of pixels
        total_pixels += image.shape[0] * image.shape[1]

    # Calculate the mean and standard deviation
    mean = sum_channel_values / total_pixels

    # Reset variables for standard deviation calculation
    sum_squared_diff = np.zeros(3)

    for i, image_file in enumerate(image_files):
        print(i)
        image = cv2.imread(os.path.join(folder_path, image_file))
        image = image.astype(np.float32) / 255.0

        # Calculate squared differences from the mean
        squared_diff = np.square(image - mean)

        # Sum squared differences for each channel
        sum_squared_diff += np.sum(squared_diff, axis=(0, 1))

    # Calculate the standard deviation
    std = np.sqrt(sum_squared_diff / total_pixels)

    return mean, std

if __name__ == "__main__":
    folder_path = "/home/tigran/Downloads/tasks/squirrels_head"
    mean, std = calculate_mean_and_std(folder_path)
    print(f"Mean (R, G, B): {mean}")
    print(f"Standard Deviation (R, G, B): {std}")
