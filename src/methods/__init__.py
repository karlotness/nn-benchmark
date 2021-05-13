from sklearn import neighbors
from . import hnn, srnn, mlp, nn_kernel, cnn

def build_network(net_args):
    arch = net_args["arch"]
    arch_args = net_args["arch_args"]
    if arch == "hnn":
        return hnn.build_network(arch_args)
    elif arch == "srnn":
        return srnn.build_network(arch_args)
    elif arch == "mlp-deriv":
        return mlp.build_network(arch_args, predict_type="deriv")
    elif arch == "mlp-step":
        return mlp.build_network(arch_args, predict_type="step")
    elif arch == "cnn-deriv":
        return cnn.build_network(arch_args, predict_type="deriv")
    elif arch == "cnn-step":
        return cnn.build_network(arch_args, predict_type="step")
    elif arch in {"knn-regressor", "knn-predictor",
                  "knn-regressor-oneshot", "knn-predictor-oneshot"}:
        return neighbors.KNeighborsRegressor(n_neighbors=1)
    elif arch == "hogn":
        # Lazy import HOGN to avoid pytorch-geometric if possible
        from . import hogn
        return hogn.build_network(arch_args)
    elif arch == "gn":
        # Lazy import GN to avoid pytorch-geometric if possible
        from . import gn
        return gn.build_network(arch_args)
    elif arch == "nn-kernel-deriv":
        return nn_kernel.build_network(arch_args, predict_type="deriv")
    elif arch == "nn-kernel-step":
        return nn_kernel.build_network(arch_args, predict_type="step")
    else:
        raise ValueError(f"Invalid architecture: {arch}")
