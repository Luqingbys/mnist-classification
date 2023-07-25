import torch


class Vae():
    def __init__(self, model_path: str, device='cpu') -> None:
        self.model = torch.load(model_path, map_location=device)
    
    def encode(self, x):
        enc = self.model.encode(x)
        return self.model.reparameterize(enc).detach().cpu().numpy()


class Ae():
    def __init__(self, model_path: str, device='cpu') -> None:
        self.model = torch.load(model_path, map_location=device)

    def encode(self, x):
        enc: torch.Tensor = self.model.encode(x)
        return enc.detach().cpu().numpy()