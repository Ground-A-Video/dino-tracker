import subprocess
import argparse
import os


INTERVAL_EROSION_DICT = {
    20: [10, 20, 40],
    30: [10, 30, 40],
    35: [30],
    40: [20, 25, 35, 45],
    50: [10, 20]
}


def run_commands(root_path):

    # Infer all data paths
    data_paths = [
        os.path.join(root_path, vid_name)
        for vid_name in os.listdir(root_path)
    ]
    
    # Global loop: data_path
    for data_path in sorted(data_paths):
        commands = [
            [
                "python", "./preprocessing/main_preprocessing.py",
                "--config", "./config/preprocessing_small.yaml",
                "--data-path", data_path
            ],

            [
                "python", "./train.py",
                "--config", "./config/zeroscope/train_640_w_refiner_250.yaml",
                "--data-path", data_path
            ]
        ]
        
        for command in commands:
            try:
                print(f"Running command: {' '.join(command)}")
                subprocess.run(command)
            except subprocess.CalledProcessError as e:
                print(f"Error during inference for {data_path}, interval {interval}: {e}")
                continue


        for iteration in range(10, 360, 10):

            for interval, kernel_sizes in INTERVAL_EROSION_DICT.items():

                try:
                    # Command 2: Inference with segmentation mask (run once per interval)
                    inference_command = [
                        "python", "inference_grid.py",
                        "--config", "config/zeroscope/train_640_w_refiner_250.yaml",
                        "--data-path", data_path,
                        "--use-segm-mask",
                        "--interval", str(interval),
                        "--iter", str(iteration),
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
                            "--iter", str(iteration),
                            "--point-size", "60",
                        ]
                        subprocess.run(visualize_command, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Error during visualization for {data_path}, interval {interval}, kernel size {kernel_size}: {e}")
                        continue



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple commands for training, inference, and visualization.")
    parser.add_argument("root_path", type=str, help="The root path containing video data.")
    args = parser.parse_args()

    run_commands(args.root_path)



'''
[24]
python train_then_visualize_under_400.py dataset/BADJA_24_under_400/BADJA_original;

python train_then_visualize_under_400.py dataset/BADJA_24_under_400/BADJA_ours_40k;


python train_then_visualize_under_400.py dataset/davis_24_under_400/davis_original;

python train_then_visualize_under_400.py dataset/davis_24_under_400/davis_ours_40k;


[full]
python train_then_visualize_under_400.py dataset/BADJA_under_400/BADJA_ours_40k;
python train_then_visualize_under_400.py dataset/davis_under_400/davis_ours_40k;

python train_then_visualize_under_400.py dataset/BADJA_under_400/BADJA_original;
python train_then_visualize_under_400.py dataset/davis_under_400/davis_original;

'''