import argparse

from scripts.config import load_config
from scripts.pipeline import run

def parse_args():
    p = argparse.ArgumentParser(description="CoLD: Collaborative Label Denoising")
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--results_dir", type=str, default=None)
    p.add_argument("--n_classes", type=int, default=None)
    p.add_argument("--noise_type", type=str, default=None, choices=["sys", "asys"])
    p.add_argument("--noise_rate", type=float, default=None)
    p.add_argument("--n_subsets", type=int, default=None)
    p.add_argument("--overlap", type=float, default=None)
    p.add_argument("--pretrain_epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--classifier", type=str, default=None, choices=["xgboost", "logistic", "mlp"])
    p.add_argument("--cluster_method", type=str, default=None, choices=["kmeans", "gmm"])
    p.add_argument("--encoder_checkpoint", type=str, default=None)
    p.add_argument("--force_pretrain", action="store_true")
    p.add_argument("--no_save_encoder", action="store_true")
    p.add_argument("--no_load_encoder", action="store_true")
    p.add_argument("--no_purify", action="store_true")
    p.add_argument("--no_reorder", action="store_true")
    p.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    return p.parse_args()

def main():
    args = parse_args()
    skip = {
        "config", "force_pretrain", "no_save_encoder", "no_load_encoder",
        "no_purify", "no_reorder",
    }
    overrides = {k: v for k, v in vars(args).items() if k not in skip and v is not None}
    if args.no_purify:
        overrides["purify"] = False
    if args.no_reorder:
        overrides["feature_reordering"] = False
    if args.force_pretrain:
        overrides["force_pretrain"] = True
    if args.no_save_encoder:
        overrides["save_encoder"] = False
    if args.no_load_encoder:
        overrides["load_encoder_if_exists"] = False

    config = load_config(args.config, **overrides)
    run(config)

if __name__ == "__main__":
    main()
