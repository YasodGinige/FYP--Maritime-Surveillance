import cv2
import os

def create_video_from_images(image_folder, video_name, fps=30):
    # Get a list of image files in the folder
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    # Sort the image files
    image_files.sort()

    # Read the first image to get its dimensions
    first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = first_image.shape

    # Create a VideoWriter object to write the video
    video_writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Loop through the image files, resize them and write them to the video
    for image_file in image_files:
        img = cv2.imread(os.path.join(image_folder, image_file))
        resized_img = cv2.resize(img, (width, height))
        video_writer.write(resized_img)

    # Release the video writer and close all windows
    video_writer.release()
    cv2.destroyAllWindows()

# Example usage
image_folder = "/home/fyp3-2/Desktop/BATCH18/YOWO/output/"
video_name = '/home/fyp3-2/Desktop/BATCH18/YOWO/output_video/ht1.mp4'
fps = 30

create_video_from_images(image_folder, video_name, fps)
