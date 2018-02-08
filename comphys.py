"""Executable for Computational Physics exercises

Command line args syntax:
    [assignment name in lowercase][exercise][subexercise]
    Example: comphys excitons1b
"""
from assignments.shooting import *
import sys

if __name__ == "__main__":
    eval(sys.argv[1]+"()")
