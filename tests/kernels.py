"""Definition of kernels to be used throughout testing"""

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel


kernels = [RBF(length_scale=1.0),
           C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0),
           C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)]


def get_kernel():
    try:
        index = 0
        while True:
            try:
                value = (yield kernels[index])
                index += 1
            except Exception as e:
                value = e
