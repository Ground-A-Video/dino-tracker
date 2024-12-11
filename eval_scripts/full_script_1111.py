import subprocess
import argparse
import os

def run_commands(root_data_path, dataset_name, is_24=False):
    data_paths = [os.path.join(root_data_path, d) for d in os.listdir(root_data_path) if os.path.isdir(os.path.join(root_data_path, d))]
    
    # for data_path in sorted(data_paths):
    #     commands = [
    #         [
    #             "python", "./preprocessing/main_preprocessing.py",
    #             "--config", "./config/preprocessing_small.yaml",
    #             "--data-path", data_path
    #         ],

    #         # [
    #         #     "python", "./train.py",
    #         #     "--config", "./config/zeroscope/train_640.yaml",
    #         #     "--data-path", data_path
    #         # ]
    #     ]
        
    #     for command in commands:
    #         print(f"Running command: {' '.join(command)}")
    #         subprocess.run(command)


    # Run additional commands after all data paths are processed
    for iteration in [1000, 2000, 5000, 10000]:
    # for iteration in range(1):
        if is_24:
            additional_commands = [
                [
                    "python", "eval_scripts/inference.py",
                    "--config-path", "config/zeroscope/train_640_w_refiner.yaml",
                    "--root-dataset-path", root_data_path,
                    "--dataset-name", dataset_name,
                    "--iter", str(iteration),
                    "--is_24",
                ],

                [
                    "python", "eval_scripts/evaluate.py",
                    "--root-dataset-path", root_data_path,
                    "--dataset-name", dataset_name,
                    "--iter", str(iteration),
                    "--is_24",
                ]
            ]

        else:
            additional_commands = [
                [
                    "python", "eval_scripts/inference.py",
                    "--config-path", "config/zeroscope/train_640_w_refiner.yaml",
                    "--root-dataset-path", root_data_path,
                    "--dataset-name", dataset_name,
                    # "--iter", str(iteration)
                ],

                [
                    "python", "eval_scripts/evaluate.py",
                    "--root-dataset-path", root_data_path,
                    "--dataset-name", dataset_name,
                    "--iter", str(iteration)
                ]
            ]
        
        for command in additional_commands:
            print(f"Running command: {' '.join(command)}")
            subprocess.run(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run preprocessing and training scripts for all data paths in the root directory.")
    parser.add_argument("--root-data-path", required=True, help="Root path of the dataset")
    parser.add_argument("--dataset-name", required=True, choices=["davis", "BADJA"], help="Dataset name to process (either 'davis' or 'BADJA')")
    parser.add_argument("--is_24", action="store_true")
    args = parser.parse_args()
    
    run_commands(args.root_data_path, args.dataset_name, args.is_24)



'''
[BADJA_24]
python eval_scripts/full_script.py --root-data-path dataset/BADJA_24/BADJA_ours_25k --dataset-name BADJA --is_24
python eval_scripts/full_script.py --root-data-path dataset/BADJA_24/BADJA_ours_30k --dataset-name BADJA --is_24
python eval_scripts/full_script.py --root-data-path dataset/BADJA_24/BADJA_ours_35k --dataset-name BADJA --is_24
python eval_scripts/full_script.py --root-data-path dataset/BADJA_24/BADJA_ours_40k --dataset-name BADJA --is_24

python eval_scripts/full_script.py --root-data-path dataset/BADJA_24/BADJA_svd_original_big_hack --dataset-name BADJA --is_24
python eval_scripts/full_script.py --root-data-path dataset/BADJA_24/BADJA_svd_original_hack --dataset-name BADJA --is_24


[davis_24]
python eval_scripts/full_script.py --root-data-path dataset/davis_24/davis_ours_25k --dataset-name davis --is_24
python eval_scripts/full_script.py --root-data-path dataset/davis_24/davis_ours_30k --dataset-name davis --is_24
python eval_scripts/full_script.py --root-data-path dataset/davis_24/davis_ours_35k --dataset-name davis --is_24
python eval_scripts/full_script.py --root-data-path dataset/davis_24/davis_ours_40k --dataset-name davis --is_24

python eval_scripts/full_script.py --root-data-path dataset/davis_24/davis_svd_original_big_hack --dataset-name davis --is_24
python eval_scripts/full_script.py --root-data-path dataset/davis_24/davis_svd_original_hack --dataset-name davis --is_24


[BADJA]
python eval_scripts/full_script.py --root-data-path dataset/BADJA/BADJA_ours_25k --dataset-name BADJA
python eval_scripts/full_script.py --root-data-path dataset/BADJA/BADJA_ours_30k --dataset-name BADJA
python eval_scripts/full_script.py --root-data-path dataset/BADJA/BADJA_ours_35k --dataset-name BADJA
python eval_scripts/full_script.py --root-data-path dataset/BADJA/BADJA_ours_40k --dataset-name BADJA

python eval_scripts/full_script.py --root-data-path dataset/BADJA/BADJA_svd_original_big_hack --dataset-name BADJA
python eval_scripts/full_script.py --root-data-path dataset/BADJA/BADJA_svd_original_hack --dataset-name BADJA


[davis]
python eval_scripts/full_script.py --root-data-path dataset/davis/davis_ours_25k --dataset-name davis
python eval_scripts/full_script.py --root-data-path dataset/davis/davis_ours_30k --dataset-name davis
python eval_scripts/full_script.py --root-data-path dataset/davis/davis_ours_35k --dataset-name davis 
python eval_scripts/full_script.py --root-data-path dataset/davis/davis_ours_40k --dataset-name davis 

python eval_scripts/full_script.py --root-data-path dataset/davis/davis_svd_original_big_hack --dataset-name davis 
python eval_scripts/full_script.py --root-data-path dataset/davis/davis_svd_original_hack --dataset-name davis 


'''