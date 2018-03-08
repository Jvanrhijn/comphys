"""Executable for Computational Physics exercises

Command line args syntax:
    [assignment name in lowercase][exercise][subexercise]
    Example: comphys excitons1b
"""
import sys
from assignments.excitons import *
from assignments.monte_carlo import *
import assignments.excitons as excitons
import assignments.monte_carlo as monte_carlo


usage = """
Usage:
    $ python comphys.py [assignment name][task number][sub task letter]
    
Example:
    $ python comphys.py excitons4b         
"""

if __name__ == "__main__":
    first_argument = sys.argv[1]
    valid_arguments = dir(excitons) + dir(monte_carlo)
    if "-h" in sys.argv:
        print("Valid arguments:")
        for item in valid_arguments:
            if 'monte_carlo' in item or 'excitons' in item:
                print('    '+item)
        exit(0)
    if first_argument in valid_arguments:
        eval(first_argument + "()")
        exit(0)
    else:
        print(usage)
        exit(1)
