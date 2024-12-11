import subprocess
import argparse


def run_benchmark(config_path, root_dataset_path, dataset_name, iteration, is_24=False):
    if dataset_name == "davis":
        video_ids = range(30)
        if is_24:
            benchmark_pickle_path = "tapvid/tapvid_davis_data_strided_24.pkl"
        else:
            benchmark_pickle_path = "tapvid/tapvid_davis_data_strided.pkl"

    elif dataset_name == "BADJA":
        video_ids = range(9)
        if is_24:
            benchmark_pickle_path = "tapvid/tapvid_BADJA_data_24.pkl"
        else:
            benchmark_pickle_path = "tapvid/tapvid_BADJA_data.pkl"
    else:
        raise ValueError("Invalid dataset name. Must be either 'davis' or 'BADJA'.")
    
    for video_id in video_ids:
        data_path = f"{root_dataset_path}/{video_id}"
        if iteration is not None:
            command = [
                "python", "inference_benchmark.py",
                "--config", config_path,
                "--benchmark-pickle-path", benchmark_pickle_path,
                "--data-path", data_path,
                "--video-id", str(video_id),
                "--iter", str(iteration)
            ]
        else:
            command = [
            "python", "inference_benchmark.py",
            "--config", config_path,
            "--benchmark-pickle-path", benchmark_pickle_path,
            "--data-path", data_path,
            "--video-id", str(video_id),
        ]
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference_benchmark.py for a specific dataset.")
    parser.add_argument("--config-path", required=True, help="Path to the configuration file")
    parser.add_argument("--root-dataset-path", required=True, help="Root path of the dataset")
    parser.add_argument("--dataset-name", choices=["davis", "BADJA"], required=True, help="Dataset name to process (either 'davis' or 'BADJA')")
    parser.add_argument("--iter", type=int, default=None)
    parser.add_argument("--is_24", action="store_true")
    args = parser.parse_args()
    
    run_benchmark(args.config_path, args.root_dataset_path, args.dataset_name, args.iter, args.is_24)



'''
[BADJA]
(dino_w_refiner)
python eval_scripts/inference.py --config-path config/train_dino_1_w_refiner.yaml --root-dataset-path dataset/BADJA/BADJA_dino_w_refiner --dataset-name BADJA

(dino_wo_refiner)
python eval_scripts/inference.py --config-path config/train_dino_1_wo_refiner.yaml --root-dataset-path dataset/BADJA_24/BADJA_dino_wo_refiner --dataset-name BADJA

(svd_original_hack) (wo_refiner)
python eval_scripts/inference.py --config-path config/train_svd_1_wo_refiner.yaml --root-dataset-path dataset/BADJA/BADJA_svd_original_hack --dataset-name BADJA

(svd_original_zero) (wo_refiner)
python eval_scripts/inference.py --config-path config/train_svd_1_wo_refiner.yaml --root-dataset-path dataset/BADJA/BADJA_svd_original_zero --dataset-name BADJA

(svd_original_big_hack) (wo_refiner)
python eval_scripts/inference.py --config-path config/train_svd_1_wo_refiner.yaml --root-dataset-path dataset/BADJA/BADJA_svd_original_big_hack --dataset-name BADJA

(svd_original_big_zero) (wo_refiner)
python eval_scripts/inference.py --config-path config/train_svd_1_wo_refiner.yaml --root-dataset-path dataset/BADJA/BADJA_svd_original_big_zero --dataset-name BADJA

(ours 40k)
python eval_scripts/inference.py --config-path config/train_svd_1_wo_refiner.yaml --root-dataset-path dataset/BADJA_24/BADJA_ours_40k --dataset-name BADJA

(ours 25k)
python eval_scripts/inference.py --config-path config/train_svd_1_wo_refiner.yaml --root-dataset-path dataset/BADJA_24/BADJA_ours_25k --dataset-name BADJA --iter 0

[davis]
(dino_w_refiner)
python eval_scripts/inference.py --config-path config/train_dino_1_w_refiner.yaml --root-dataset-path dataset/davis/davis_dino_w_refiner --dataset-name davis

(dino_wo_refiner)
python eval_scripts/inference.py --config-path config/train_dino_1_wo_refiner.yaml --root-dataset-path dataset/davis_24/davis_dino_wo_refiner --dataset-name davis --is_24

(svd_original_hack) (wo_refiner)
python eval_scripts/inference.py --config-path config/train_svd_1_wo_refiner.yaml --root-dataset-path dataset/davis/davis_svd_original_hack --dataset-name davis

(svd_original_zero) (wo_refiner)
python eval_scripts/inference.py --config-path config/train_svd_1_wo_refiner.yaml --root-dataset-path dataset/davis/davis_svd_original_zero --dataset-name davis

(svd_original_big_hack) (wo_refiner)
python eval_scripts/inference.py --config-path config/train_svd_1_wo_refiner.yaml --root-dataset-path dataset/davis/davis_svd_original_big_hack --dataset-name davis

(svd_original_big_zero) (wo_refiner)
python eval_scripts/inference.py --config-path config/train_svd_1_wo_refiner.yaml --root-dataset-path dataset/davis/davis_svd_original_big_zero --dataset-name davis

'''