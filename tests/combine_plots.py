"""Plots to visually compare the predictions of the experts in an rBCM.

Should create a single static html file in tests/visuals when run. Externally
you should only call the compare_1d_plots function."""

import os
import numpy as np
from bokeh.plotting import figure, output_file, save, gridplot
from bokeh.palettes import Blues4, Spectral11


def _create_residuals_plot(X, y1, y2, model1_name=None, model2_name=None):
    """Show the difference between the predictions of two models"""
    p = figure(title="Small residuals between rBCM and GPR models' predictions")
    p.scatter(X, y1 - y2, color='black', legend="Residual between predictive means",
              alpha=0.5, line_width=3)
    p.xaxis.axis_label = "X"
    if model1_name is not None and model2_name is not None:
        p.yaxis.axis_label = "Predictive Mean of " + str(model1_name) + " - Predictive Mean of " + str(model2_name)
    else:
        p.yaxis.axis_label = "Predictive Mean of Model 1 - Predictive Mean of Model 2"
    p.line(X, 0, color='blue', line_width=5, legend="No Difference in Predictive Mean")
    return p


def _var_diff_plot(X, y1, sigma1, y2, sigma2, model1_name=None, model2_name=None):
    title = "Difference of the variance of the posteriors: Var(" + str(model1_name) + ") - Var(" + str(model2_name) + ")"
    p = figure(title=title)
    var1 = np.power(sigma1, 2)
    var2 = np.power(sigma2, 2)
    diff = var1 - var2
    p.scatter(X, diff.ravel(), color='black', line_width=3, alpha=0.5, legend="At point x: Var(GPR) - Var(rBCM)")
    p.xaxis.axis_label = "X"
    if model1_name is not None and model2_name is not None:
        p.yaxis.axis_label = "Posterior Predictive Var(" + str(model1_name) + ") - Var(" + str(model2_name) + ")"
    else:
        p.yaxis.axis_label = "Posterior Predictive Var(Model 1) - Var(Model 2)"
    p.line(X, 0, color='blue', line_width=5, legend="No Difference in Predictive Certainty")
    return p


def _create_regression_plot(X, y, predict_X, predict_y, sigma, model_name=None, time=-1, clusters=None):
    p = figure(title="The " + str(model_name) + " model - " + str(time) + "s - Kernel: C(1.0) * RBF(1.0) + WhiteKernel(1.0)", width=600, plot_height=600)

    top = predict_y + 1.96 * sigma
    bot = predict_y - 1.96 * sigma
    left = predict_X - (1.0 / 2) * (predict_X[1] - predict_X[0])
    right = predict_X + (1.0 / 2) * (predict_X[1] - predict_X[0])
    p.quad(top=top.ravel(), bottom=bot.ravel(), left=left.ravel(),
           right=right.ravel(), color=Blues4[1],
           alpha=0.5, line_width=0, legend="95% CI Band")
    p.line(predict_X, predict_y, color='black', legend='Predictive Mean', line_width=3)
    if clusters is None:
        p.scatter(X, y, size=3, legend="Underlying dataset", color='black')
    else:
        if len(clusters) == 1:
            p.scatter(X[clusters[0]], y[clusters[0]], size=3,
                      legend="Underlying dataset", color='black')
        else:
            for i in range(len(clusters)):
                p.scatter(X[clusters[i]], y[clusters[i]], size=3, color=Spectral11[2 * i % len(Spectral11)],
                          legend="Underlying data cluster: " + str(i))
    p.xaxis.axis_label = "X"
    p.yaxis.axis_label = "y"
    return p


def compare_1d_plots(output_filename, X, y, predict_X,
                     gpr_y, gpr_sigma, gpr_time,
                     rbcm_y, rbcm_sigma, rbcm_time, expert_sets):
    """Create a scatter plot with the 1d regression lines of both the GPR and
    the rBCM through the data. Assumes both models were predicted onto the
    same points, predict_X.
    """
    if expert_sets is None:
        expert_sets = []
    # If we don't have both times don't show either
    if rbcm_time is None or gpr_time is None:
        rbcm_time = "Unknown "
        gpr_time = "Unknown "
    gpr_regression_plot = _create_regression_plot(X, y, predict_X, gpr_y, gpr_sigma, model_name="GPR", time=gpr_time)
    var_difference_plot = _var_diff_plot(predict_X, gpr_y, gpr_sigma, rbcm_y, rbcm_sigma, model1_name="GPR", model2_name="rBCM")
    residuals_plot = _create_residuals_plot(predict_X, gpr_y, rbcm_y)
    rbcm_regression_plot = _create_regression_plot(X, y, predict_X, rbcm_y, rbcm_sigma, model_name="rBCM", time=rbcm_time, clusters=expert_sets)

    title = "GPR took: " + str(gpr_time) + "s     rBCM took: " + str(rbcm_time) + "s"

    output_file(os.path.dirname(__file__) + "/visuals/" + output_filename + ".html", title=title)

    grid = gridplot([[gpr_regression_plot, rbcm_regression_plot],
                     [residuals_plot, var_difference_plot]])

    save(grid)
