from typing import AnyStr, Callable
import torch

# class Extractor:
#     def __init__(self) -> None:
#         pass

# class DurationEtractor(Extractor):
#     def extract

def decode_durations(enc_duration: int, sampling_rate:int = 22050, hop_length:int = 256  ) -> float:
    return enc_duration*hop_length/sampling_rate


# def extract_duration()

def extra_extract(extract_func: Callable, idx:int):
    
    def wrapper(func: Callable, batch: torch.Tensor, *args, **kwargs):
        rounded_durations = batch[idx]
        dec_durations = [decode_durations(du) for du in rounded_durations]
        res = func(batch, *args, **kwargs)

        return res

    return wrapper
