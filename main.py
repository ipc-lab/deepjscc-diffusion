import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb

# from runners.diffusion import Diffusion
from guided_diffusion.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# CUDA_VISIBLE_DEVICES=1 python main.py --config imagenet_512.yml --path_y celeba_hq --eta 0.85 --deg "sr_averagepooling" --deg_scale 2 --sigma_y 0.1 --target /home/sfy21/deepjscc-local/logs/diffjscc_eval_final/multiruns/2023-09-06_08-01-27/0/ --ni --batch_size 12
# CUDA_VISIBLE_DEVICES=1 python main.py --config imagenet_512.yml --path_y celeba_hq --eta 0.85 --deg "sr_averagepooling" --deg_scale 2 --sigma_y 0.1 --target /home/sfy21/deepjscc-local/logs/diffjscc_eval_final/multiruns/2023-09-06_08-02-37/0/ --ni --batch_size 12

# scp -r mavka.dept.ic.ac.uk:/home/sfy21/deepjscc-local/logs/diffjscc_eval_final/multiruns/2023-09-06_08-01-27 . && scp -r mavka.dept.ic.ac.uk:/home/sfy21/deepjscc-local/logs/diffjscc_eval_final/multiruns/2023-09-06_08-02-37 .
# sshfs mavka.dept.ic.ac.uk:/home/sfy21/DDNM_new DDNM_new

# kaira0: CUDA_VISIBLE_DEVICES=0 python main.py --config imagenet_512.yml --path_y celeba_hq --eta 0.85 --deg "sr_averagepooling" --deg_scale 2 --sigma_y 0.1 --target /home/sfy21/deepjscc-local/logs/diffjscc_eval_final/multiruns/2023-09-06_08-02-37/2/ --ni --batch_size 12 && CUDA_VISIBLE_DEVICES=0 python main.py --config imagenet_512.yml --path_y celeba_hq --eta 0.85 --deg "sr_averagepooling" --deg_scale 2 --sigma_y 0.1 --target /home/sfy21/deepjscc-local/logs/diffjscc_eval_final/multiruns/2023-09-06_08-02-37/3/ --ni --batch_size 12
# kaira1: CUDA_VISIBLE_DEVICES=1 python main.py --config imagenet_512.yml --path_y celeba_hq --eta 0.85 --deg "sr_averagepooling" --deg_scale 2 --sigma_y 0.1 --target /home/sfy21/deepjscc-local/logs/diffjscc_eval_final/multiruns/2023-09-06_08-02-37/4/ --ni --batch_size 12 && CUDA_VISIBLE_DEVICES=1 python main.py --config imagenet_512.yml --path_y celeba_hq --eta 0.85 --deg "sr_averagepooling" --deg_scale 2 --sigma_y 0.1 --target /home/sfy21/deepjscc-local/logs/diffjscc_eval_final/multiruns/2023-09-06_08-02-37/5/ --ni --batch_size 12

# draeyah1: CUDA_VISIBLE_DEVICES=1 python main.py --config imagenet_512.yml --path_y celeba_hq --eta 0.85 --deg "sr_averagepooling" --deg_scale 2 --sigma_y 0.1 --target /home/sfy21/deepjscc-local/logs/diffjscc_eval_final/multiruns/2023-09-06_08-01-27/1/ --ni --batch_size 12 && CUDA_VISIBLE_DEVICES=1 python main.py --config imagenet_512.yml --path_y celeba_hq --eta 0.85 --deg "sr_averagepooling" --deg_scale 2 --sigma_y 0.1 --target /home/sfy21/deepjscc-local/logs/diffjscc_eval_final/multiruns/2023-09-06_08-01-27/2/ --ni --batch_size 12
# mavka 3 of 01-27
# kaira 4 and 5 of 01-27
# scp -r kaira.ee.ic.ac.uk:/home/sfy21/deepjscc-local/logs/diffjscc_eval_final /home/sfy21/deepjscc-local/logs/
# scp -r iblis.ee.ic.ac.uk:/home/sfy21/deepjscc-local/logs/diffjscc_eval_final /home/sfy21/deepjscc-local/logs/
# scp -r draeyah.ee.ic.ac.uk:/home/sfy21/deepjscc-local/logs/diffjscc_eval_final /home/sfy21/deepjscc-local/logs/
# scp -r bouloulou.ee.ic.ac.uk:/home/sfy21/deepjscc-local/logs/diffjscc_eval_final /home/sfy21/deepjscc-local/logs/
# scp -r moxina.ee.ic.ac.uk:/home/sfy21/deepjscc-local/logs/diffjscc_eval_final /home/sfy21/deepjscc-local/logs/


# draeyah0: CUDA_VISIBLE_DEVICES=0 python main.py --config imagenet_512.yml --path_y celeba_hq --eta 0.85 --deg "sr_averagepooling" --deg_scale 2 --sigma_y 0.0 --target /home/sfy21/deepjscc-local/logs/diffjscc_eval_final/multiruns/2023-09-06_08-02-37/0/ --ni --batch_size 12 && CUDA_VISIBLE_DEVICES=0 python main.py --config imagenet_512.yml --path_y celeba_hq --eta 0.85 --deg "sr_averagepooling" --deg_scale 2 --sigma_y 0.0 --target /home/sfy21/deepjscc-local/logs/diffjscc_eval_final/multiruns/2023-09-06_08-02-37/0/ --ni --batch_size 12

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Set different seeds for diverse results")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--deg", type=str, required=True, help="Degradation"
    )
    parser.add_argument(
        "--path_y",
        type=str,
        required=True,
        help="Path of the test dataset.",
    )
    parser.add_argument(
        "--sigma_y", type=float, default=0., help="sigma_y"
    )
    parser.add_argument(
        "--eta", type=float, default=0.85, help="Eta"
    )    
    parser.add_argument(
        "--simplified",
        action="store_true",
        help="Use simplified DDNM, without SVD",
    )    
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--deg_scale", type=float, default=0., help="deg_scale"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        '--subset_start', type=int, default=-1
    )
    parser.add_argument(
        '--subset_end', type=int, default=-1
    )
    parser.add_argument(
        '--batch_size', type=int, default=16
    )
    parser.add_argument(
        "-n",
        "--noise_type",
        type=str,
        default="gaussian",
        help="gaussian | 3d_gaussian | poisson | speckle"
    )
    parser.add_argument(
        "--add_noise",
        action="store_true"
    )

    

    args = parser.parse_args()

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)



    os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
    args.image_folder = os.path.join(
        args.target, "image_samples_" + str(args.sigma_y) + ("_simplified" if args.simplified else "") + "_ffhq", args.image_folder # + (f"_subset{args.subset_start}-{args.subset_end}" if args.subset_start >= 0 else "")
    )
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input(
                f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
            )
            if response.upper() == "Y":
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder, exist_ok=True)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    try:
        runner = Diffusion(args, config)
        runner.sample(args.simplified)
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
