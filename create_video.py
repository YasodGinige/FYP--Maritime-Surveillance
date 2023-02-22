import cv2
import os

# Define the path to the folder containing the image frames
path = '/home/fyp3-2/Desktop/BATCH18/YOWO/output/'

# Define the frame rate of the output video
fps = 20

# Define the size of the output video frame
width, height = (448, 448)

# Get the list of image file names in the specified folder
files = os.listdir(path)
files.sort()

# Create a VideoWriter object to write the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# Loop through each image file in the folder and add it to the output video
for file in files:
    if file.endswith('.jpg') or file.endswith('.png'):
        # Load the image file
        img = cv2.imread(os.path.join(path, file))
        
        # Resize the image to match the output video frame size
        img = cv2.resize(img, (width, height))
        
        # Add the image to the output video
        out.write(img)

# Release the VideoWriter object
out.release()
