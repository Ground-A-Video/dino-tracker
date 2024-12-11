import os
import glob


def keep_first_24_frames(base_path):
    # Navigate through all video-name folders in the base path
    for video_folder in os.listdir(base_path):
        video_path = os.path.join(base_path, video_folder)
        if not os.path.isdir(video_path):
            continue

        # Process both 'masks' and 'video' subfolders
        for subfolder in ['masks', 'video']:
            subfolder_path = os.path.join(video_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            # Get all image files (.jpg and .png) sorted by name
            image_files = sorted(glob.glob(os.path.join(subfolder_path, '*.jpg')) +
                                glob.glob(os.path.join(subfolder_path, '*.png')))

            # Keep only the first 24 frames, delete the rest
            for image_file in image_files[24:]:
                os.remove(image_file)
                print(f"Deleted: {image_file}")

if __name__ == "__main__":
    base_directory = "./dataset/davis_24/davis_ours"
    keep_first_24_frames(base_directory)
