import subprocess
import argparse
import os


# Define global variable for interval and erosion kernel size pairs
INTERVAL_EROSION_DICT = {
    15: [20, 25, 30],
    20: [20, 25, 30],
    25: [20, 25],
    30: [20, 25],
    40: [20, 25],
}

# Define video names and seeds
VID_NAMES = [
    "a-bald-eagle-flying-over-a-tree-filled-forest",
    "a-blue-fishing-boat-is-navigating-in-the-ocean-next-to-a-cruise-ship",
    "a-blue-train-traveling-through-a-lush-green-area",
    "a-brown-and-white-cow-eating-hay",
    "a-bunch-of-food-is-cooking-on-a-grill-over-an-open-fire",
    "a-butterfly-sits-on-top-of-a-purple-flower",
    "a-city-bus-driving-down-a-snowy-street-at-night",
    "a-close-up-of-a-piece-of-sushi-on-chopsticks",
    "a-close-up-of-leaves-with-water-droplets-on-them",
    "a-great-white-shark-swimming-in-the-ocean",
    "a-green-toy-car-is-sitting-on-the-ground",
    "a-hand-holding-a-slice-of-pizza",
    "a-lioness-yawning-in-a-field"
]
SEEDS = [1, 12, 123, 1234, 12345, 123456, 1234567]

def run_commands(root_path):
    # Infer all data paths
    data_paths = [
        os.path.join(root_path, "576_320", vid_name, f"seed_{seed}")
        for vid_name in VID_NAMES for seed in SEEDS
    ]

    # Loop through all data paths
    for data_path in data_paths:
        try:
        # Command 1: Preprocessing
            preprocess_command = [
                "python", "./preprocessing/main_preprocessing.py",
                "--config", "./config/preprocessing.yaml",
                "--data-path", data_path
            ]
            subprocess.run(preprocess_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during preprocessing for {data_path}: {e}")
            continue

        # Loop through given intervals and erosion kernel sizes
        for interval, kernel_sizes in INTERVAL_EROSION_DICT.items():
            try:
                # Command 2: Inference with segmentation mask (run once per interval)
                inference_command = [
                    "python", "inference_grid.py",
                    "--config", "config/zeroscope/train_640.yaml",
                    "--data-path", data_path,
                    "--use-segm-mask",
                    "--interval", str(interval)
                ]
                subprocess.run(inference_command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error during inference for {data_path}, interval {interval}: {e}")
                continue
            
            # Command 3: Visualization (run for each erosion kernel size)
            for kernel_size in kernel_sizes:
                try:
                    visualize_command = [
                        "python", "visualization/visualize_rainbow.py",
                        "--data-path", data_path,
                        "--plot-trails",
                        "--interval", str(interval),
                        "--erosion-kernel-size", str(kernel_size),
                        "--infer-res-size", "320", "576",
                        "--of-res-size", "320", "576"
                    ]
                    subprocess.run(visualize_command, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error during visualization for {data_path}, interval {interval}, kernel size {kernel_size}: {e}")
                    continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple commands for preprocessing, inference, and visualization.")
    parser.add_argument("root_path", type=str, help="The root path containing video data.")
    args = parser.parse_args()

    run_commands(args.root_path)


'''
python generated_track_script.py /home/work/hyeonho-video/repos/SVD-Tracker/generated_vid_track/svd_original_25
python generated_track_script.py /home/work/hyeonho-video/repos/SVD-Tracker/generated_vid_track/svd_original_30
python generated_track_script.py /home/work/hyeonho-video/repos/SVD-Tracker/generated_vid_track/svd_original_40
python generated_track_script.py /home/work/hyeonho-video/repos/SVD-Tracker/generated_vid_track/svd_original_50
python generated_track_script.py /home/work/hyeonho-video/repos/SVD-Tracker/generated_vid_track/ours_davis_jump_40k_25
python generated_track_script.py /home/work/hyeonho-video/repos/SVD-Tracker/generated_vid_track/ours_davis_jump_40k_30
python generated_track_script.py /home/work/hyeonho-video/repos/SVD-Tracker/generated_vid_track/ours_davis_jump_40k_40
python generated_track_script.py /home/work/hyeonho-video/repos/SVD-Tracker/generated_vid_track/ours_davis_jump_40k_50

'''