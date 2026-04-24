"""Utility functions for the autograder."""

from typing import Any

import torch


def save_student_data(
    student_data: dict[str, dict[str, Any]], outpath_student_data: str
):
    """Saves the student_data dict (containing information required by the autograder) to disk.
    Serializes the dict via `torch.save()`.
    Use `torch.load(path)` to deserialize it from disk.

    Args:
        student_data (dict[str, dict[str, Any]]): Student data. Main keys are:
            ["output"][str key] -> student_val.
            Example:
                ["output"]["q1_num_model_params"] -> int num_model_params
        outpath_student_data (str): Where to save the student data.
    """
    torch.save(
        {"output": student_data["output"]},
        outpath_student_data,
    )
    print(
        f"Wrote student data (with keys={student_data.keys()}) to: {outpath_student_data}"
    )
    if "output" in student_data:
        print(f"keys in 'output': {student_data['output'].keys()}")
    else:
        print(
            "Warning: No 'output' key in student data! Is your submission dict correctly populated?"
        )


def rel_error(x, y):
    return torch.max(
        torch.abs(x - y)
        / (torch.maximum(torch.tensor(1e-8), torch.abs(x) + torch.abs(y)))
    ).item()


def check_error(name, x, y, tol=1e-2):
    error = rel_error(x, y)
    if error > tol:
        print(f"The relative error for {name} is {error}, should be smaller than {tol}")
    else:
        print(f"The relative error for {name} is {error}")


def check_acc(acc, threshold):
    if acc < threshold:
        print(f"The accuracy {acc} should >= threshold accuracy {threshold}")
    else:
        print(f"The accuracy {acc} is better than threshold accuracy {threshold}")
