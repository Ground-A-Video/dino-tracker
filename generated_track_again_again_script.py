import subprocess
import argparse
import os


# Define global variable for interval and erosion kernel size pairs
INTERVAL_EROSION_DICT = {
    20: [10, 20, 30],
    30: [10, 20, 30, 40],
    35: [30],
    40: [10, 20, 30],
}


# Define video names and seeds
VID_NAMES = [
    # "a-group-of-hot-air-balloons-flying-over-a-valley",
    # "tangerines-in-a-metal-bowl-on-a-table",
    # "metal-balls-are-suspended-in-the-air",
    # "an-old-rusty-car-sits-in-the-middle-of-a-field",
    # "an-older-woman-is-drinking-a-bottle-of-water",
    # "an-airplane-is-flying-through-the-sky-at-sunset",
    # "two-sheep-grazing-in-the-grass-next-to-a-wooden-bridge",

    "a-red-bus-driving-down-a-snowy-street-at-night",
    "a-red-panda-eating-bamboo-in-a-zoo",
    "a-snow-man-holding-a-lantern-in-the-snow",
    "a-sea-turtle-swimming-in-the-ocean-under-the-water",
    "A-red-sports-car-driving-through-sand,-kicking-up-a-large-amount-of-dust",
    "A-yellow-boat-is-cruising-in-front-of-a-bridge",
    "a-blue-car-driving-down-a-dirt-road-near-train-tracks",
    "a-close-up-of-a-hippopotamus-eating-grass-in-a-field",
    "a-large-rhino-grazing-in-the-grass-near-a-bush",
    "a-space-shuttle-taking-off-into-the-sky",
    "a-train-traveling-down-tracks-through-the-woods-with-leaves-on-the-ground",
    "a-white-and-blue-airplane-flying-in-the-sky",
    "a-white-car-is-swiftly-driving-on-a-dirt-road-near-a-bush,-kicking-up-dust",
]
# SEEDS = [
#     1, 12, 123, 1234, 12345, 123456, 1234567,
#     10, 12, 1230, 12340, 123450, 1234560, 12345670,
#     11, 121, 1231, 12341, 123451, 1234561, 12345671,
#     12, 122, 1232, 12342, 123452, 1234562, 12345672,
# ]
SEEDS = [1, 12, 123, 1234, 12345, 123456, 1234567]
SEEDS = list(dict.fromkeys(SEEDS))

def run_commands(root_path):
    # Infer all data paths
    data_paths = [
        os.path.join(root_path, "576_320", vid_name, f"seed_{seed}")
        for vid_name in VID_NAMES for seed in SEEDS
    ]

    # Loop through all data paths
    for data_path in data_paths:

        if not os.path.exists(data_path):
            continue

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
                        "--of-res-size", "320", "576",
                        "--point-size", "60",
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

python generated_track_again_again_script.py /home/work/hyeonho-video/repos/SVD-Tracker/generated_vid_track_again_again/ours_davis_jump_20k_30
python generated_track_again_again_script.py /home/work/hyeonho-video/repos/SVD-Tracker/generated_vid_track_again_again/ours_davis_jump_20k_40
python generated_track_again_again_script.py /home/work/hyeonho-video/repos/SVD-Tracker/generated_vid_track_again_again/ours_davis_jump_20k_50

python generated_track_again_again_script.py /home/work/hyeonho-video/repos/SVD-Tracker/generated_vid_track_again_again/ours_davis_jump_30k_30
python generated_track_again_again_script.py /home/work/hyeonho-video/repos/SVD-Tracker/generated_vid_track_again_again/ours_davis_jump_30k_40
python generated_track_again_again_script.py /home/work/hyeonho-video/repos/SVD-Tracker/generated_vid_track_again_again/ours_davis_jump_30k_50

python generated_track_again_again_script.py /home/work/hyeonho-video/repos/SVD-Tracker/generated_vid_track_again_again/ours_davis_jump_40k_30
python generated_track_again_again_script.py /home/work/hyeonho-video/repos/SVD-Tracker/generated_vid_track_again_again/ours_davis_jump_40k_40
python generated_track_again_again_script.py /home/work/hyeonho-video/repos/SVD-Tracker/generated_vid_track_again_again/ours_davis_jump_40k_50


'''