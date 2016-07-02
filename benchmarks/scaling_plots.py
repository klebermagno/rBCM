"""Tools to create visual comparisons to Gaussian Process Regression."""
import os
import numpy as np
from bokeh.plotting import figure, output_file, save, gridplot
from bokeh.palettes import Blues4


def create_rbcm_scaling_plot():
    pass


def create_gpr_scaling_plot():
    pass


def compare_scaling_plots(output_filename, X, gpr_perf, rbcm_perf, action="fitting", partitions=1):
    """Generate and save graphs showing how the scalabiliy of gpr and rbcm differ.
    Parameters
    ----------
    gpr_perf : array, shape = (num_samples, n_observations)
        Array containing the times for fitting or predicting

    rbcm_perf : array, shape = (num_samples, n_observations)
        Array containing the times for fitting or predicting

    action : string (default : "fitting")
        Must be one of "fitting" or "predicting" to indicate if we are comparing
        time to fit or time to predict.
    """
    action = action.strip().lower()
    if action == "fit":
        pass
    elif action == "predict":
        pass
    else:
        raise ValueError("Action must be one of 'fitting' or 'predicting'")
    output_file(os.path.dirname(__file__) + "/visuals/" + output_filename + "_" + str(partitions) + ".html", title=action)
    gpr_scaling_plot = create_gpr_scaling_plot()
    rbcm_scaling_plot = create_rbcm_scaling_plot()
    grid = gridplot([[gpr_scaling_plot, rbcm_scaling_plot]])
    save(grid)
