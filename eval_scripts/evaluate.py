import subprocess
import os


def run_eval_benchmark(root_dataset_path, dataset_name, n=0, is_24=False):
    if dataset_name == "davis":
        if is_24:
            benchmark_pickle_path = "tapvid/tapvid_davis_data_strided_24.pkl"
        else:
            benchmark_pickle_path = "tapvid/tapvid_davis_data_strided.pkl"
        dataset_type = "tapvid"
        
    elif dataset_name == "BADJA":
        if is_24:
            benchmark_pickle_path = "tapvid/tapvid_BADJA_data_24.pkl"
        else:
            benchmark_pickle_path = "tapvid/tapvid_BADJA_data.pkl"
        dataset_type = "BADJA"
    else:
        raise ValueError("Invalid dataset name. Must be either 'davis' or 'BADJA'.")
    
    last_folder_name = os.path.basename(os.path.normpath(root_dataset_path))
    os.makedirs(f"tapvid/{dataset_name}", exist_ok=True)
    if is_24:
        out_file = f"tapvid/{dataset_name}_24/comp_metrics_{last_folder_name}_{n}.csv"
    else:
        out_file = f"tapvid/{dataset_name}/comp_metrics_{last_folder_name}_{n}.csv"
    
    command = [
        "python", "./eval/eval_benchmark.py",
        "--dataset-root-dir", root_dataset_path,
        "--benchmark-pickle-path", benchmark_pickle_path,
        "--out-file", out_file,
        "--dataset-type", dataset_type
    ]
    
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run eval_benchmark.py for a specific dataset.")
    parser.add_argument("--root-dataset-path", required=True, help="Root path of the dataset")
    parser.add_argument("--dataset-name", choices=["davis", "BADJA"], required=True, help="Dataset name to process (either 'davis' or 'BADJA')")
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--is_24", action="store_true")
    args = parser.parse_args()
    
    run_eval_benchmark(args.root_dataset_path, args.dataset_name, args.iter, args.is_24)


'''
[BADJA]
(dino_w_refiner)
python eval_scripts/evaluate.py --root-dataset-path dataset/BADJA/BADJA_dino_w_refiner --dataset-name BADJA

(dino_wo_refiner)
python eval_scripts/evaluate.py --root-dataset-path dataset/BADJA_24/BADJA_dino_wo_refiner --dataset-name BADJA

(svd_original_hack) (wo_refiner)
python eval_scripts/evaluate.py --root-dataset-path dataset/BADJA/BADJA_svd_original_hack --dataset-name BADJA

(svd_original_zero) (wo_refiner)
python eval_scripts/evaluate.py --root-dataset-path dataset/BADJA/BADJA_svd_original_zero --dataset-name BADJA

(svd_original_big_hack) (wo_refiner)
python eval_scripts/evaluate.py --root-dataset-path dataset/BADJA/BADJA_svd_original_big_hack --dataset-name BADJA

(svd_original_big_zero) (wo_refiner)
python eval_scripts/evaluate.py --root-dataset-path dataset/BADJA/BADJA_svd_original_big_zero --dataset-name BADJA

(ours 40k) (wo_refiner)
python eval_scripts/evaluate.py --root-dataset-path dataset/BADJA_24/BADJA_ours_40k --dataset-name BADJA

(ours 25k) (wo_refiner)
python eval_scripts/evaluate.py --root-dataset-path dataset/BADJA_24/BADJA_ours_25k --dataset-name BADJA --iter 0


[davis]
(dino_w_refiner)
python eval_scripts/evaluate.py --root-dataset-path dataset/davis/davis_dino_w_refiner --dataset-name davis

(dino_wo_refiner)
python eval_scripts/evaluate.py --root-dataset-path dataset/davis/davis_dino_wo_refiner --dataset-name davis
python eval_scripts/evaluate.py --root-dataset-path dataset/davis_24/davis_dino_wo_refiner --dataset-name davis --is_24

(svd_original_hack) (wo_refiner)
python eval_scripts/evaluate.py --root-dataset-path dataset/davis/davis_svd_original_hack --dataset-name davis

(svd_original_zero) (wo_refiner)
python eval_scripts/evaluate.py --root-dataset-path dataset/davis/davis_svd_original_zero --dataset-name davis

(svd_original_big_hack) (wo_refiner)
python eval_scripts/evaluate.py --root-dataset-path dataset/davis/davis_svd_original_big_hack --dataset-name davis

(svd_original_big_zero) (wo_refiner)
python eval_scripts/evaluate.py --root-dataset-path dataset/davis/davis_svd_original_big_zero --dataset-name davis

'''