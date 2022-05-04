import argparse
import onnx
import subprocess
import os
import logging
import pandas as pd
from datetime import datetime
from pprint import pprint
from tqdm import tqdm
import itertools


def check_onnx_model(onnx_file):
    model = onnx.load(onnx_file)
    onnx.checker.check_model(model)


def parse_progress_line(line: str) -> dict:
    field_sep = ','
    label_sep = ':'
    time_sep = ';'
    result = {}
    # Example line: "progress: 100/None, correct:  11/100, verified: 11/11, unsafe: 0/11,  time: 0.027; 0.050; 0.551"

    def numerator(s):
        """Helper function to extract numerator from a string fraction"""
        return int(s.split('/')[0].strip())

    chunks = line.split(field_sep)
    for chunk in chunks:
        label, data = map(str.strip, chunk.split(label_sep))
        if label in ["correct", "verified", "unsafe"]:
            result[label] = numerator(data)
        elif label == "progress":
            result["count"] = numerator(data)
        elif label == "time":
            times = map(float, map(str.strip, data.split(time_sep)))
            result["image_time"], result["cum_verif_time_per_verif"], result["cum_verif_time"] = times
        else:
            logging.warning(f"Unrecognized label ({label}) in progress line: {line}")
    return result


def parse_verification_result(output) -> dict:
    last_progress_line = ""
    for line in output:
        if line.startswith("progress"):
            last_progress_line = line
    result = parse_progress_line(last_progress_line)
    return result


def run_experiment(model_filepath: str, epsilon: float, eran_path="ERAN", conda_env=None) -> dict:
    # Currently only added support for the deeppoly domain. We only plot against MNIST in our report.
    dataset = "mnist"
    domain = "deeppoly"

    experiment_result = {
        "model": model_filepath,
        "epsilon": epsilon
    }

    # The file should be a valid onnx model
    try:
        check_onnx_model(model_filepath)
    except onnx.checker.ValidationError as e:
        logging.error(f"Invalid onnx model: {e}")
        return experiment_result

    eran_verify_base = os.path.join(eran_path, "tf_verify")
    eran_verify_file = "__main__.py"

    absolute_model_filepath = os.path.join(os.getcwd(), model_filepath)

    command_env = ["conda", "run", "-n", conda_env] if conda_env is not None else []
    command = [
        *command_env,
        "python", eran_verify_file,
        "--netname", absolute_model_filepath,
        "--epsilon", str(epsilon),
        "--dataset", dataset,
        "--domain", domain,
    ]
    # We run the verification as a subprocess and parse the output since the tf_verify/__main__.py file contains a lot
    # of necessary logic/preprocessing and is not set up to be called from other scripts.
    result = subprocess.run(command, capture_output=True, cwd=eran_verify_base)
    if result.returncode != 0:
        logging.error(
            f"Verification process failed with return code {result.returncode}.\n"
            f"Command was: {' '.join(command)}\n"
            f"Stderr was: {result.stderr.decode('utf-8')}\n"
        )
    else:
        experiment_result = {
            **parse_verification_result(result.stdout.decode('utf-8').splitlines()),
            **experiment_result
        }

    return experiment_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ONNX mnist model sanity check')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help='Saved ONNX models to run experiments for')
    parser.add_argument('--epsilons', type=float, nargs='+', required=True,
                        help='Epsilon values to run experiments for. Must be in the range [0, 1].')
    parser.add_argument('--eran_path', type=str, default='ERAN',
                        help='Path to ERAN base directory.')
    parser.add_argument('--save_dir', type=str, default='experiments',
                        help='Directory to save experiment results to.')
    parser.add_argument('--conda_env', type=str,
                        help='Conda environment to run experiment subprocess in. Defaults to no environment')

    args = parser.parse_args()

    results = []
    experiments = list(itertools.product(args.models, args.epsilons))
    for i in tqdm(range(len(experiments))):
        model, epsilon = experiments[i]
        print(f"\nRunning experiment for model: {model} and epsilon: {epsilon}")
        result = run_experiment(model, epsilon, args.eran_path, args.conda_env)
        results.append(result)
        print("Experiment result was:")
        pprint(result)

    df = pd.DataFrame(results)
    timestamp = datetime.now().isoformat(timespec='seconds')
    savepath = os.path.join(args.save_dir, f"experiment_{timestamp}.csv")
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    df.to_csv(savepath, index=False)
