from datasets import load_dataset
import random

def load_and_sample_wikitext(n : int = 100):
    """
    Load the Wikitext-103 dataset and sample 100,000 entries.
    Returns a list of sampled text entries.
    """

    dataset = load_dataset("hellaswag")

    split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_split = split_dataset["train"]
    val_split = split_dataset["test"]

    return train_split[:n], val_split[:n]

# Print first three entries to verify
if __name__ == "__main__":
    train_split, val_split = load_and_sample_wikitext()
