import os
import shutil
import cv2

def sort_images(source_dir, dest_dirs):
    """
    Sort images from source_dir into destination directories based on keyboard input.

    :param source_dir: Path to the source directory containing images.
    :param dest_dirs: Dictionary mapping keys to destination directories.
    """
    # Ensure destination directories exist
    for key, dest_dir in dest_dirs.items():
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
    
    # Walk through the source directory and process images
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Read and display the image
                image = cv2.imread(file_path)
                if image is None:  # Skip files that aren't valid images
                    print(f"Skipping invalid image: {file_path}")
                    continue
                
                cv2.imshow("Image Sorting", image)
                print(f"Processing: {file_path}")
                print("Press 1, 2, or 3 to sort the image into respective folders.")
                print("Press 'q' to quit.")

                while True:
                    key = cv2.waitKey(0) & 0xFF  # Wait for a key press
                    if key in [ord('1'), ord('2'), ord('3')]:
                        dest_folder = dest_dirs[chr(key)]
                        shutil.copy(file_path, dest_folder)
                        print(f"Copied to {dest_folder}")
                        break
                    elif key == ord('q'):
                        print("Quitting...")
                        cv2.destroyAllWindows()
                        return
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    cv2.destroyAllWindows()
    print("Sorting complete.")

# Example usage
if __name__ == "__main__":
    # Define source directory and destination folders
    source_directory = "path_to_source_folder"
    destination_directories = {
        '1': "path_to_destination_folder_1",
        '2': "path_to_destination_folder_2",
        '3': "path_to_destination_folder_3"
    }

    sort_images(source_directory, destination_directories)
