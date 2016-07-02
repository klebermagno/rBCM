"""Tools to create visual comparisons to Gaussian Process Regression."""
import os
import numpy as np
from bokeh.plotting import figure, output_file, save, gridplot
from bokeh.palettes import Blues4


def create_residuals_plot(X, y1, y2, model1_name=None, model2_name=None):
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


def var_diff_plot(X, y1, sigma1, y2, sigma2, model1_name=None, model2_name=None):
    title = "Difference of variance of the posteriors: Var(" + str(model1_name) + ") - Var(" + str(model2_name) + ")"
    p = figure(title=title)
    var1 = np.power(sigma1, 2)
    var2 = np.power(sigma2, 2)
    diff = var1 - var2
    p.scatter(X, diff.ravel(), color='black', line_width=3, alpha=0.5)
    p.xaxis.axis_label = "X"
    if model1_name is not None and model2_name is not None:
        p.yaxis.axis_label = "Posterior Predictive Var(" + str(model1_name) + ") - Var(" + str(model2_name) + ")"
    else:
        p.yaxis.axis_label = "Posterior Predictive Var(Model 1) - Var(Model 2)"
    p.line(X, 0, color='blue', line_width=5, legend="No Difference in Predictive Certainty")
    return p


def create_regression_plot(X, y, predict_X, predict_y, sigma, model_name=None, time=-1):
    p = figure(title="The " + str(model_name) + " model - " + str(time) + "s - Kernel: C(1.0) * RBF(1.0) + WhiteKernel(1.0)", width=600, plot_height=600)

    top = predict_y + 1.96 * sigma
    bot = predict_y - 1.96 * sigma
    left = predict_X - (1.0 / 2) * (predict_X[1] - predict_X[0])
    right = predict_X + (1.0 / 2) * (predict_X[1] - predict_X[0])
    p.quad(top=top.ravel(), bottom=bot.ravel(), left=left.ravel(),
           right=right.ravel(), color=Blues4[1],
           alpha=0.5, line_width=0)
    p.line(predict_X, predict_y, color='black', legend='Predictive Mean', line_width=3)
    p.scatter(X, y, size=5, legend="Underlying dataset", color='black')
    p.xaxis.axis_label = "X"
    p.yaxis.axis_label = "y"
    return p


def compare_1d_plots(output_filename, X, y, predict_X,
                     gpr_y, gpr_sigma, gpr_time,
                     rbcm_y, rbcm_sigma, rbcm_time):
    """Create a scatter plot with the 1d regression lines of both the GPR and
    the rBCM through the data. Assumes both models were predicted onto the
    same points, predict_X.
    """
    gpr_regression_plot = create_regression_plot(X, y, predict_X, gpr_y, gpr_sigma, model_name="GPR", time=gpr_time)
    var_difference_plot = var_diff_plot(predict_X, gpr_y, gpr_sigma, rbcm_y, rbcm_sigma, model1_name="GPR", model2_name="rBCM")
    residuals_plot = create_residuals_plot(predict_X, gpr_y, rbcm_y)
    rbcm_regression_plot = create_regression_plot(X, y, predict_X, rbcm_y, rbcm_sigma, model_name="rBCM", time=rbcm_time)

    title = "GPR took: " + str(gpr_time) + "s     rBCM took: " + str(rbcm_time) + "s"

    output_file(os.path.dirname(__file__) + "/visuals/" + output_filename + ".html", title=title)

    grid = gridplot([[gpr_regression_plot, rbcm_regression_plot],
                     [residuals_plot, var_difference_plot]])

    save(grid)
