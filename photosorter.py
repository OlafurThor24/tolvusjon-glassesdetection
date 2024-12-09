import os
import shutil
import cv2

def sort_images(source_dir, dest_dirs):
    """
    Sort images from source_dir into destination directories based on keyboard input.

    :param source_dir: Path to the source directory containing images.
    :param dest_dirs: Dictionary mapping keys to lists of destination directories.
    """
    # Ensure all destination directories exist
    for key, dest_list in dest_dirs.items():
        for dest_dir in dest_list:
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
                print("Press 1: Circular, 2: Square, 3: Both, or 4: Neither to sort the image into respective folders.")
                print("Press 'q' to quit.")

                while True:
                    key = cv2.waitKey(0) & 0xFF  # Wait for a key press
                    if key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                        # Copy the file to all directories associated with the pressed key
                        for dest_folder in dest_dirs[chr(key)]:
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
    source_directory = "classification/eyeglasses/face-attributes-extra/test/eyeglasses"
    destination_directories = {
        '1': [
            "classification/circularglasses/face-attributes-extra/test/circularglasses",
            "classification/squareglasses/face-attributes-extra/test/no_squareglasses"
        ],
        '2': [
            "classification/squareglasses/face-attributes-extra/test/squareglasses",
            "classification/circularglasses/face-attributes-extra/test/no_circularglasses"
        ],
        '3': [
            "classification/circularglasses/face-attributes-extra/test/circularglasses",
            "classification/squareglasses/face-attributes-extra/test/squareglasses"
        ],
        '4': [
            "classification/squareglasses/face-attributes-extra/test/no_squareglasses",
            "classification/circularglasses/face-attributes-extra/test/no_circularglasses"
        ]
    }

    sort_images(source_directory, destination_directories)
