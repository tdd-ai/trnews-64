from .utils import *

try:
    from .utils_torch import load_model, nll
    print("PyTorch is installed, running with PyTorch.")
except ImportError:
    print("PyTorch is not installed.")
    try:
        from .utils_flax import load_model, nll
        print("Flax is installed, running with Flax.")
    except ImportError:
        print("Flax is not installed.")
        raise ImportError("Neither Flax nor PyTorch is installed.")
