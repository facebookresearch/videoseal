import io
import locale
import sys
from typing import Any, Dict

import torch
import triton_python_backend_utils as pb_utils  # type: ignore


def _fix_locale() -> None:
    # Docker containers default to ASCII.
    locale.setlocale(locale.LC_ALL, "C.UTF-8")
    # If certain pieces of code print unicode characters, we need
    # to ensure both stdout and stderr are configured to do so.
    # Otherwise we'll get an encoding error at runtime
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8")


class TritonPythonModelBase:
    def initialize(self, args):
        _fix_locale()
        self.device = (
            torch.device("cuda", int(args["model_instance_device_id"]))
            if args["model_instance_kind"] == "GPU"
            else torch.device("cpu")
        )

    def get_singleton_input(self, request, name: str) -> Any:
        tensor = pb_utils.get_input_tensor_by_name(
            request,
            name,
        )
        if tensor is None:
            return None
        tensor = tensor.as_numpy()
        assert tensor.shape == (1,)
        return tensor[0]
