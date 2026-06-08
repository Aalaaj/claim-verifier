"""Training modules for fine-tuning models"""

from .dataset import (
    download_scifact,
    prepare_claim_detection_data,
    prepare_verifier_data
)
from .train_claim_detector import train_claim_detector
from .train_verifier import train_verifier

__all__ = [
    "download_scifact",
    "prepare_claim_detection_data",
    "prepare_verifier_data",
    "train_claim_detector",
    "train_verifier",
]