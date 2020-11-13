""" A file that contains general functions useful for programming with GPUs.
"""

import subprocess as sp
import os


def get_gpu_memory(verbose=0):
    """Reports available memory in MB of each visible GPU on the system.
    """
    # Function to parse console output
    def _output_to_list(x): return x.decode('ascii').split('\n')[:-1]

    # Console command to be executed
    command = "nvidia-smi --query-gpu=memory.free --format=csv"

    # Clean and parse console output
    free_memory_info = _output_to_list(sp.check_output(command.split()))[1:]
    free_memory_values = [int(memory.split()[0])
                          for i, memory in enumerate(free_memory_info)]
    if verbose > 0:
        pretty_vals = [mem_val/1000 for mem_val in free_memory_values]
        print(pretty_vals)
    return free_memory_values


def useable_gpu_memory(buff=256):
    """ Gives safely usable memory to avoid graphical failure
    Args:
        buff (int, optional): Memory buffer so GPU isn't maxed. Defaults to 256 MB.
    Returns:
        list[int]: List of buffered GPU memory
    """
    gpu_memory = get_gpu_memory()
    return [int(memory-buff) for memory in gpu_memory]
