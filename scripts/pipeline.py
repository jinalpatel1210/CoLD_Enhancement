import os
import numpy as np
import torch

from .classify import train_classifier, train_and_evaluate
from .denoise import Denoiser, recover_candidates
from .data import DataModule
from .encoder import EncoderTrainer
from .metrics import noise_detection_metrics
from .utils import ensure_dir, set_seed

def run(config):
    set_seed(config.seed)
    run_dir = ensure_dir(os.path.join(config.results_dir, config.run_name))

    line = "=" * 78
    print(f"\n{line}\nCoLD run: {config.run_name}\n{line}")

    dm = DataModule(config).setup()
    config.save(os.path.join(run_dir, "config.yaml"))

    print(f"\n{line}\n[1] representation learning\n{line}")
    ckpt = config.resolve_encoder_checkpoint_path()
    if config.load_encoder_if_exists and not config.force_pretrain and os.path.isfile(ckpt):
        print(f"[pretrain] checkpoint found: {ckpt}")
    encoder = EncoderTrainer(config, dm).fit_or_load()

    set_seed(config.eval_seed)
    torch.manual_seed(config.eval_seed)

    z_train, z_views, y_noisy, y_clean = encoder.embed(dm.train_loader(shuffle=False))
    z_test, _, y_test, _ = encoder.embed(dm.test_loader())

    results = {}

    if config.purify:
        print(f"\n{line}\n[2] label denoising ({config.cluster_method})\n{line}")
        keep_mask, denoise_info = Denoiser(config).purify(z_train, z_views, y_noisy, y_clean)
        original_keep = keep_mask.copy()

        print(f"\n{line}\n[3] Train classifier on purified samples\n{line}")

        clf = train_classifier(
            config,
            z_train[keep_mask],
            y_noisy[keep_mask]
        )
    
        cluster_labels = denoise_info["cluster_labels"]

        agreement_labels = denoise_info["agreement_labels"]

        candidate_indices, agreement_labels = recover_candidates(
    original_keep,
    cluster_labels,
)

        ############################################################
        # Classifier prediction on Agreement=1 candidates
        ############################################################

        if len(candidate_indices) > 0:

            candidate_prediction = clf.predict(
                z_train[candidate_indices]
            )

            recover_mask = (
                candidate_prediction == agreement_labels
            )

            recovered_indices = candidate_indices[recover_mask]

            recovered_labels = agreement_labels[recover_mask]

        else:

            recover_mask = np.array([], dtype=bool)

            recovered_indices = np.array([], dtype=int)

            recovered_labels = np.array([], dtype=int)

        ############################################################
        # Randomly keep only MAX_RECOVER samples
        ############################################################

        MAX_RECOVER = config.max_recover_samples

        if len(recovered_indices) > MAX_RECOVER:

            rng = np.random.default_rng(config.seed)

            selected = rng.choice(
                len(recovered_indices),
                size=MAX_RECOVER,
                replace=False
            )

            recovered_indices = recovered_indices[selected]

            recovered_labels = recovered_labels[selected]

        final_keep = keep_mask.copy()

        final_keep[recovered_indices] = True

        final_labels = y_noisy.copy()

        final_labels[recovered_indices] = recovered_labels

        print("========= Original CoLD =========")

        print(f"Kept by CoLD               : {original_keep.sum()}")

        print(f"Discarded by CoLD          : {(~original_keep).sum()}")

        print()

        print("========= Recovery =========")

        print(f"Agreement=1 Candidates     : {len(candidate_indices)}")

        print(f"Classifier Matches         : {recover_mask.sum()}")

        print(f"Recovered (after sampling) : {len(recovered_indices)}")

        print()

        print("========= Final Dataset =========")

        print(f"Training Samples           : {final_keep.sum()}")

        print(f"Discarded Samples          : {(~final_keep).sum()}")

        print(f"\n{line}\n[4] Agreement-based recovery\n{line}")
        metrics, _, _ = train_and_evaluate(
    config,
    z_train[final_keep],
    final_labels[final_keep],
    z_test,
    y_test,
)
        print(f"[eval] test macro-F1 = {metrics['macro_f1']:.4f}")
        results["eval"] = metrics
    else:
        print(f"\n{line}\n[5] Final downstream evaluation\n{line}")
        metrics, _, _ = train_and_evaluate(
            config,
            z_train,
            y_noisy,
            z_test,
            y_test,
        )
        print(f"[eval] test macro-F1 = {metrics['macro_f1']:.4f}")
        results["eval"] = metrics

    print(f"\n{line}\nDONE  macro-F1={results['eval']['macro_f1']:.4f}\n{line}")
    return results
