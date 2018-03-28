"""Executable for Computational Physics exercises

Command line args syntax:
    [assignment name in lowercase][exercise][subexercise]
    Example: comphys excitons1b
"""
import sys
from assignments.excitons import *
from assignments.monte_carlo import *
from assignments.wave_propagation import *
from assignments.molecular_dynamics import *


usage = """
Usage:
    $ python comphys.py [assignment name][task number][sub task letter]
    
Example:
    $ python comphys.py excitons4b         
"""


def is_valid_argument(arg):
    is_assignment = "monte_carlo" in arg or "excitons" in arg or "wave_propagation" in arg \
                    or "molecular_dynamics" in arg
    exists = arg in all_functions
    return is_assignment and exists


if __name__ == "__main__":
    first_argument = sys.argv[1]
    global all_functions
    all_functions = dir()
    if "-h" in sys.argv:
        print("Valid arguments:")
        for item in all_functions:
            if is_valid_argument(item):
                print('    '+item)
        exit(0)
    if is_valid_argument(first_argument):
        eval(first_argument + "()")
        exit(0)
    else:
        print(usage)
        exit(1)
