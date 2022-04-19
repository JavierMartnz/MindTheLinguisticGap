import argparse
# sys.path.append('/vol/research/SignProduction/Code/Tao_Code/sign_spotting/exp1_BSLCP')
from src.train_i3d import train
# from test import test
# from evaluate import evaluate
# from evaluate_bbc import evaluate_bbc

# tj = :  command : python3 -m src train ./config.yaml
def main():
    ap = argparse.ArgumentParser("LFTS reprodibility experiment")

    ap.add_argument("--mode", choices=["train", "test", "eval", "eval_bbc"], default="train",
                    help="train a model")

    ap.add_argument("--config_path", type=str, default="configs/config.yaml",
                    help="path to YAML config file")

    ap.add_argument("--ckpt", type=str,
                    help="checkpoint for prediction")

    ap.add_argument("--seq_name", type=str)

    ap.add_argument("--sub_dir", type=str)

    ap.add_argument("--output_dir", type=str)


    args = ap.parse_args()

    if args.mode == "train":
        train(cfg_file=args.config_path)
    # elif args.mode == "test":
    #     test(cfg_file=args.config_path, ckpt=args.ckpt)
    # elif args.mode == "eval":
    #     evaluate(cfg_file=args.config_path, ckpt=args.ckpt)
    # elif args.mode == "eval_bbc":
    #     evaluate_bbc(cfg_file=args.config_path, ckpt=args.ckpt)
    else:
        raise ValueError("unknown mode")

if __name__ == "__main__":
    main()
