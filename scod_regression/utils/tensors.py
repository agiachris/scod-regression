from typing import Optional, TypeVar, Union, Callable, Tuple, List, Type

import numpy as np
import torch


Scalar = Union[np.generic, float, int, bool]
scalars = (np.generic, float, int, bool)
T = TypeVar("T")


def flatten(x: Union[Tuple[T, ...], List[T]]) -> List[T]:
    """Flatten Iterable of Iterables recursively."""
    if not x:
        return list(x)
    if isinstance(x[0], (tuple, list)):
        return flatten(x[0]) + flatten(x[1:])
    return list(x[:1]) + flatten(x[1:])


def device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Return torch.device from x."""
    if isinstance(device, torch.device):
        return device
    elif device is None:
        return torch.device("cpu")
    elif device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def to(structure, device: torch.device):
    """Moves the nested structure to the given device.

    Numpy arrays are converted to Torch tensors first.

    Args:
        structure: Nested structure.
        device: Torch device.

    Returns:
        Transferred structure.
    """

    def _to(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        else:
            return torch.from_numpy(x).to(device)

    return map_structure(_to, structure)


def map_structure(
    func: Callable,
    *args,
    atom_type: Union[Type, Tuple[Type, ...]] = (torch.Tensor, np.ndarray),
):
    """Maps the function over the structure containing either Torch tensors or Numpy
    arrays.

    Args:
        func: Function to be mapped.
        *args: Nested structure arguments of `func`.
        atom_type: Type to which the function should be applied.
    """
    return nest_map_structure(
        func,
        *args,
        atom_type=atom_type,
        skip_type=(np.ndarray, torch.Tensor, *scalars, str, type(None)),
    )


def nest_map_structure(
    func: Callable,
    *args,
    atom_type: Union[Type, Tuple[Type, ...]] = (
        torch.Tensor,
        np.ndarray,
        *scalars,
        type(None),
    ),
    skip_type: Optional[Union[Type, Tuple[Type, ...]]] = None,
):
    """Applies the function over the nested structure atoms.

    Works like tensorflow.nest.map_structure():
    https://www.tensorflow.org/api_docs/python/tf/nest/map_structure

    Args:
        func: Function applied to the atoms of *args.
        *args: Nested structure arguments of `func`.
        atom_type: Types considered to be atoms in the nested structure.
        skip_type: Types to be skipped and returned as-is in the nested structure.

    Returns:
        Results of func(*args_atoms) in the same nested structure as *args.
    """
    arg_0 = args[0]
    if isinstance(arg_0, atom_type):
        return func(*args)
    elif skip_type is not None and isinstance(arg_0, skip_type):
        return arg_0 if len(args) == 1 else args
    elif isinstance(arg_0, dict):
        return {
            key: nest_map_structure(
                func,
                *(arg[key] for arg in args),
                atom_type=atom_type,
                skip_type=skip_type,
            )
            for key in arg_0
        }
    elif hasattr(arg_0, "__iter__"):
        iterable_class = type(arg_0)
        return iterable_class(
            nest_map_structure(func, *args_i, atom_type=atom_type, skip_type=skip_type)
            for args_i in zip(*args)
        )
    else:
        return arg_0 if len(args) == 1 else args
