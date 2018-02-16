"""Executable for Computational Physics exercises

Command line args syntax:
    [assignment name in lowercase][exercise][subexercise]
    Example: comphys excitons1b
"""
import sys
import assignments.excitons as assignments


usage = """
Usage:
    $ python comphys.py [assignment name][task number][sub task letter]
    
Example:
    $ python comphys.py excitons4b         
"""

if __name__ == "__main__":
    argument = sys.argv[1]
    if argument in dir(assignments):
        eval("assignments." + argument + "()")
        exit(0)
    else:
        print(usage)
        exit(1)
