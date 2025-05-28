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