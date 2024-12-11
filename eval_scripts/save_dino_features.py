import subprocess
import argparse

def run_command(dataset_name, root_dataset_path):
    base_command = [
        "python", "./preprocessing/save_dino_embed_video.py",
        "--config", "./config/preprocessing.yaml",
        "--data-path"
    ]
    
    if dataset_name == "davis":
        video_ids = range(30)
    elif dataset_name == "BADJA":
        video_ids = range(9)
    else:
        raise ValueError("Invalid dataset name. Must be either 'davis' or 'BADJA'.")
    
    for video_id in video_ids:
        data_path = f"{root_dataset_path}/{video_id}"
        command = base_command + [data_path]
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run save_dino_embed_video.py for a specific dataset.")
    parser.add_argument("--dataset-name", choices=["davis", "BADJA"], help="Dataset name to process (either 'davis' or 'BADJA')")
    parser.add_argument("--root-dataset-path", help="Root path of the dataset")
    args = parser.parse_args()
    
    run_command(args.dataset_name, args.root_dataset_path)


'''
[BADJA]
python eval_scripts/save_dino_features.py --dataset-name BADJA --root-dataset-path dataset/BADJA/BADJA_dino_w_refiner
python eval_scripts/save_dino_features.py --dataset-name BADJA --root-dataset-path dataset/BADJA/BADJA_dino_wo_refiner
python eval_scripts/save_dino_features.py --dataset-name BADJA --root-dataset-path dataset/BADJA_24/BADJA_dino_wo_refiner


[davis]
python eval_scripts/save_dino_features.py --dataset-name davis --root-dataset-path dataset/davis/davis_dino_w_refiner
python eval_scripts/save_dino_features.py --dataset-name davis --root-dataset-path dataset/davis/davis_dino_wo_refiner
python eval_scripts/save_dino_features.py --dataset-name davis --root-dataset-path dataset/davis_24/davis_dino_wo_refiner


'''