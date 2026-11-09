from __future__ import annotations

from typing import Union

import torch


def select_device(requested: Union[str, int, None] = None) -> str:
    if requested is not None and str(requested).strip() != "":
        return str(requested)
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_label(device: str) -> str:
    if device == "cpu":
        return "cpu"
    if device == "mps":
        return "mps"
    if str(device).isdigit() and torch.cuda.is_available():
        return f"cuda:{device} ({torch.cuda.get_device_name(int(device))})"
    return str(device)


def synchronize(device: str) -> None:
    if str(device).isdigit() and torch.cuda.is_available():
        torch.cuda.synchronize()
