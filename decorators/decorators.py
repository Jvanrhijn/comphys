"""Useful decorators, to keep assignments and helper functions clean

Decorators are a way to modify function behavior without modifying the function body.
Python has a very nice syntax for decorators, known as the 'pie' syntax (PEP 318)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)


def plot_grid_show(assignment):
    """Show all plots and show grids on the axes"""
    def wrapper():
        ax = assignment()
        if type(ax) == np.ndarray:
            for axis in ax:
                axis.grid()
        else:
            ax.grid()
        plt.show()
    return wrapper


def vertical_subplots(assignment):
    """Plot two vertical subplots, sharing a common x-axis"""
    def wrapper():
        fig, ax = plt.subplots(2, sharex=True)
        x, y_upper, y_lower, x_label, y_label_upper, y_label_lower = assignment()
        ax[0].plot(x, y_upper, color='#1f77b4')
        ax[1].set_xlabel(x_label)
        ax[0].set_ylabel(y_label_upper)
        ax[1].set_ylabel(y_label_lower)
        ax[1].plot(x, y_lower, color='#ff7f0e')
        ax[0].grid(), ax[1].grid()
        plt.show()
    return wrapper


def single_plot(assignment):
    """Plot a single function"""
    def wrapper():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x, y, x_label, y_label = assignment()
        ax.plot(x, y, color='#1f77b4')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid()
        plt.show()
    return wrapper


def plot_single_window(*args):
    """Plot multiple lines in a single window"""
    def decorator(assignment):
        def wrapper(**kwargs):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            x, y_arrays, plot_labels = assignment(**kwargs)
            for y, label in zip(y_arrays, plot_labels):
                ax.plot(x, y, label=label)
            ax.set_xlabel(args[0])
            ax.set_ylabel(args[1])
            ax.grid()
            ax.legend()
            plt.show()
        return wrapper
    return decorator
