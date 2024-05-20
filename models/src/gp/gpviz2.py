import numpy as np
import matplotlib.pyplot as plt
import torch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Tuple, List, Optional

from util.fetchdevice import fetch_device
