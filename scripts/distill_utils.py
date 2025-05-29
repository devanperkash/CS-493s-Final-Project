import torch

def get_device():
    if torch.backends.mps.is_available():
        print("✅ Using Apple MPS (Metal) backend")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("✅ Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("⚠️ Using CPU only")
        return torch.device("cpu")
    
def count_parameters(model):
    try:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
    except:
        print("Could not count model parameters.")