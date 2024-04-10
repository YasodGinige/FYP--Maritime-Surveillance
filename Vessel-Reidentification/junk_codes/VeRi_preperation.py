import os
import random
import shutil

def arrange_images_with_samples(VeriPath):
    # Create a directory to store the organized folders
    output_folder = os.path.join(VeriPath, 'organized_images')
    src_img_path = os.path.join(VeriPath, )
    folder_type = ['train', 'query', 'test']

    for OrgType in folder_type:
        if OrgType == 'query' or OrgType == 'test':
            Type = 'gallery'
        else:
            Type = OrgType
        output_directory = os.path.join(output_folder, Type)

        os.makedirs(output_directory, exist_ok=True)
        input_file = os.path.join(VeriPath, 'name_' + OrgType + '.txt')
        # Read the input file
        with open(input_file, 'r') as file:
            image_names = file.readlines()
        id_folder_list = []
        # Organize images into folders
        for image_name in image_names:
            # Extract the ID number from the image name
            id_number = image_name.split('_')[0]

            # Create a folder for the ID number if it doesn't exist
            id_folder = os.path.join(output_directory, id_number)
            os.makedirs(id_folder, exist_ok=True)
            if id_number not in id_folder_list:
                id_folder_list.append(id_number)
            # Copy the image to the corresponding folder
            image_path = os.path.join(VeriPath, 'image_'+OrgType +'/' + image_name.strip())
            shutil.copy(image_path, id_folder)
        for id_folder_no in id_folder_list:
            if OrgType != 'query':
                # Create a new folder for samples
                if Type == 'gallery':
                    samples = 'query'
                if Type == 'train':
                    samples = 'valid'
                samples_folder = os.path.join(output_folder, samples+'/' + id_folder_no)
                os.makedirs(samples_folder, exist_ok=True)

                # Get a list of images in the ID folder
                id_folder = os.path.join(output_directory, id_folder_no)
                images_in_folder = os.listdir(id_folder)

                # Select 2-3 random sample images
                sample_images = random.sample(images_in_folder, k=min(3, len(images_in_folder)))

                # Move the sample images to the samples folder
                for sample_image in sample_images:
                    sample_image_path = os.path.join(id_folder, sample_image)
                    shutil.move(sample_image_path, samples_folder)

                print("Images have been arranged into folders based on ID number.")
                print("Each folder now contains a 'samples' folder with 2-3 sample images.")
    folder_names = ['gallery','query','train','valid']
    for folder_name in folder_names:
        rename_subfolders(os.path.join(output_folder,folder_name))

def rename_subfolders(folder_path):
    # Get the list of subfolders in the folder
    subfolders = next(os.walk(folder_path))[1]
    num_subfolders = len(subfolders)

    # Calculate the maximum length of the renamed subfolders
    max_length = len(str(num_subfolders - 1))

    # Rename the subfolders
    for i, subfolder in enumerate(subfolders):
        new_name = str(i).zfill(max_length)  # Pad the number with leading zeros
        subfolder_path = os.path.join(folder_path, subfolder)
        new_subfolder_path = os.path.join(folder_path, new_name)
        os.rename(subfolder_path, new_subfolder_path)

if __name__ == '__main__':
    arrange_images_with_samples(VeriPath='/home/fyp3/Desktop/Batch18/Re_ID/VeRi')