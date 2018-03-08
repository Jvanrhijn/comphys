# Computational Physics

This repository contains the solutions to the University of Twente Computational Physics course of Applied Physics. Helper functions are implemented in different files under /lib/, while scripts are found in /comphys.py.

**Dependencies**

Python 3, numpy, matplotlib, and tqdm (for progress bars in loops). All numerical routines are implemented in /lib/*, so numpy is really only used for array handling.

**Usage**

Just call comphys.py with first argument the name of the project + assignment_number + sub_assignment_letter. Example:

`python comphys.py excitons1c`

Calling with the `-h` flag lists all valid arguments.