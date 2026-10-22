"""Vision and network helpers shared by the re_book scripts.

The robot sees the world through a small camera on its core. A frame is
reduced to how much green (the target) is in its left, middle and right
thirds, and a small fully connected network turns that, with the robot's own
state, into one target angle per hinge. Scripts 1 (all three variants), 4 and
5 import these; each script runs from its own folder, so ``import _vision``
finds this file.
"""

# Third-party libraries
import cv2
import numpy as np
import torch
from torch import nn


class Network(nn.Module):
    """Two ELU hidden layers and a tanh output scaled to plus or minus pi/2."""

    def __init__(
        self, input_size: int, output_size: int, hidden_size: int = 32
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.hidden_act = nn.ELU()
        self.out_act = nn.Tanh()
        # The weights come from an optimiser, never from gradients
        for p in self.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, model, data, state: np.ndarray) -> np.ndarray:  # noqa: ARG002
        x = torch.as_tensor(state, dtype=torch.float32)
        x = self.hidden_act(self.fc1(x))
        x = self.hidden_act(self.fc2(x))
        return (self.out_act(self.fc_out(x)) * (torch.pi / 2)).numpy()


def fill_parameters(net: nn.Module, vector: np.ndarray) -> None:
    """Copy a flat vector into the network's weights, layer by layer.

    Raises
    ------
    IndexError
        When the vector is not exactly as long as the network's weights.
    """
    address = 0
    for p in net.parameters():
        d = p.data.view(-1)
        n = len(d)
        d[:] = torch.as_tensor(vector[address : address + n])
        address += n
    if address != len(vector):
        raise IndexError("Parameter vector length mismatch")


def isolate_green(frame: np.ndarray) -> np.ndarray:
    """Return a mask of the green pixels of an RGB frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    return cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))


def analyze_sections(mask: np.ndarray) -> list[float]:
    """Split mask into left / centre / right thirds; return green fraction each."""
    sections = np.array_split(mask, 3, axis=1)
    return [cv2.countNonZero(s) / max(s.size, 1) for s in sections]
