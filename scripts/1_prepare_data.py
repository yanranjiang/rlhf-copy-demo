"""
Explore and validate the ultrafeedback_binarized dataset before training.
Run this first to confirm your dataset access and understand the data format.
"""

from datasets import load_dataset
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    data_cfg = cfg["data"]

    print(f"Loading {data_cfg['dataset_name']}...")
    ds = load_dataset(data_cfg["dataset_name"])

    print("\n--- Dataset overview ---")
    print(ds)

    train = ds[data_cfg["train_split"]]
    eval_ = ds[data_cfg["eval_split"]]
    print(f"\nTrain samples : {len(train)}")
    print(f"Eval samples  : {len(eval_)}")
    print(f"Columns       : {train.column_names}")

    sample = train[0]
    print("\n--- Sample entry ---")
    print(f"Prompt: {sample['prompt'][:300]}")
    print(f"\nChosen role: {sample['chosen'][-1]['role']}")
    print(f"Chosen text: {sample['chosen'][-1]['content'][:300]}")
    print(f"\nRejected text: {sample['rejected'][-1]['content'][:300]}")

    scores_chosen = train["score_chosen"]
    scores_rejected = train["score_rejected"]
    diffs = [c - r for c, r in zip(scores_chosen, scores_rejected)]
    mean_diff = sum(diffs) / len(diffs)

    print("\n--- Score statistics ---")
    print(f"Mean score difference (chosen - rejected): {mean_diff:.3f}")
    print(f"Min diff: {min(diffs):.3f}  |  Max diff: {max(diffs):.3f}")

    high_conf = sum(1 for d in diffs if d >= 2.0)
    print(f"High-confidence pairs (diff >= 2.0): {high_conf} ({100*high_conf/len(diffs):.1f}%)")

    print("\n--- Subsample sizes for training ---")
    print(f"Will use {data_cfg['max_train_samples']} train / {data_cfg['max_eval_samples']} eval samples")
    print("\nDataset looks good. Proceed with: python scripts/2_dpo_train.py")


if __name__ == "__main__":
    main()
