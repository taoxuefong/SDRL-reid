from __future__ import absolute_import

from .triplet import TripletLoss, SoftTripletLoss
from .crossentropy import CrossEntropyLabelSmooth, SoftEntropy
from .lces import LCes, PartLCes, CameraProxy
from .msc import MSCLoss, MSCMemory

__all__ = [
    'TripletLoss',
    'CrossEntropyLabelSmooth',
    'SoftTripletLoss',
    'SoftEntropy',
    'PartLCes',
    'LCes',
    'CameraProxy',
    'MSCLoss',
    'MSCMemory',
]