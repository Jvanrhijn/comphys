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
    argument = sys.argv[1]
    if argument in dir(excitons) + dir(monte_carlo):
        eval(argument + "()")
        exit(0)
    else:
        print(usage)
        exit(1)
