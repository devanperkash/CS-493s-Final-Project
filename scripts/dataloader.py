from datasets import load_dataset

def load_hellaswag(n_samples:int = 100, test_size:float=0.2, seed=42):
    dataset = load_dataset("hellaswag", trust_remote_code=True)

    split_dataset = dataset["train"].train_test_split(test_size=test_size, seed=seed)
    train_split = split_dataset["train"]
    val_split = split_dataset["test"]

    return train_split['ctx'][:n_samples], val_split['ctx'][:n_samples]

def load_gsm8k(n_samples:int = 100, test_size:float=0.2, seed=42):
    dataset = load_dataset("openai/gsm8k", "main")

    split_dataset = dataset["train"].train_test_split(test_size=test_size, seed=seed)
    train_split = split_dataset["train"]
    val_split = split_dataset["test"]

    return train_split['question'][:n_samples], val_split['question'][:n_samples]
