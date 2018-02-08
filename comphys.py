"""Executable for Computational Physics exercises

Command line args syntax:
    [assignment name in lowercase][exercise][subexercise]
    Example: comphys excitons1b
"""
import sys
from assignments.shooting import *

if __name__ == "__main__":
    eval(sys.argv[1]+"()")
