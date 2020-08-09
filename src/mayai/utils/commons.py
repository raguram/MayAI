import torch
import gc


def isCuda():
    return torch.cuda.is_available()


def getDevice():
    return torch.device("cuda" if isCuda() else "cpu")


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()
