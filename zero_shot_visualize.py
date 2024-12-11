import subprocess
import argparse
import os


INTERVAL_EROSION_DICT = {
    20: [10, 20, 40],
    30: [10, 30, 40],
    35: [30],
    40: [20, 25, 35, 45],
    50: [30, 45],
    60: [40]
}


def run_commands(root_path):
    # Infer all data paths
    data_paths = [
        os.path.join(root_path, vid_name)
        for vid_name in os.listdir(root_path)
    ]
    # data_paths = [ os.path.join(root_path, "28"), os.path.join(root_path, "29") ]

    # Loop through all data paths
    for data_path in data_paths:
        # try:
        # # Command 1: Preprocessing
        #     preprocess_command = [
        #         "python", "./preprocessing/main_preprocessing.py",
        #         "--config", "./config/preprocessing.yaml",
        #         "--data-path", data_path
        #     ]
        #     subprocess.run(preprocess_command, check=True)
        # except subprocess.CalledProcessError as e:
        #     print(f"Error during preprocessing for {data_path}: {e}")
        #     continue

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
[BADJA] (24)
===============================================================================
# python zero_shot_visualize.py "dataset/BADJA_24/BADJA_ours_20k"
# python zero_shot_visualize.py "dataset/BADJA_24/BADJA_ours_25k"
# python zero_shot_visualize.py "dataset/BADJA_24/BADJA_ours_30k"
# python zero_shot_visualize.py "dataset/BADJA_24/BADJA_ours_35k"
# python zero_shot_visualize.py "dataset/BADJA_24/BADJA_ours_40k"

# python zero_shot_visualize.py "dataset/BADJA_24/BADJA_original"
# python zero_shot_visualize.py "dataset/BADJA_24/BADJA_original_big"

# python zero_shot_visualize.py "dataset/BADJA_24/BADJA_wo_refiner_20k"
# python zero_shot_visualize.py "dataset/BADJA_24/BADJA_wo_corr_20k"

# python zero_shot_visualize.py "dataset/BADJA_24/BADJA_wo_refiner_25k"
# python zero_shot_visualize.py "dataset/BADJA_24/BADJA_wo_corr_25k"
===============================================================================
python zero_shot_visualize.py "dataset/BADJA_24/BADJA_zeroscope"


[davis] (24)
===============================================================================
# python zero_shot_visualize.py "dataset/davis_24/davis_ours_20k"
# python zero_shot_visualize.py "dataset/davis_24/davis_ours_25k"
# python zero_shot_visualize.py "dataset/davis_24/davis_ours_30k"
# python zero_shot_visualize.py "dataset/davis_24/davis_ours_35k"
# python zero_shot_visualize.py "dataset/davis_24/davis_ours_40k"

# python zero_shot_visualize.py "dataset/davis_24/davis_original"
# python zero_shot_visualize.py "dataset/davis_24/davis_original_big"

# python zero_shot_visualize.py "dataset/davis_24/davis_wo_refiner_20k"
# python zero_shot_visualize.py "dataset/davis_24/davis_wo_corr_20k"

# python zero_shot_visualize.py "dataset/davis_24/davis_wo_refiner_25k"
# python zero_shot_visualize.py "dataset/davis_24/davis_wo_corr_25k"
===============================================================================
python zero_shot_visualize.py "dataset/davis_24/davis_zeroscope"


'''