from sklearn import neighbors
from . import hnn, srnn, mlp

def build_network(net_args):
    arch = net_args["arch"]
    arch_args = net_args["arch_args"]
    if arch == "hnn":
        return hnn.build_network(arch_args)
    elif arch == "srnn":
        return srnn.build_network(arch_args)
    elif arch == "mlp":
        return mlp.build_network(arch_args)
    elif arch == "knn_regressor":
        return neighbors.KNeighborsRegressor(n_neighbors=5)
    else:
        raise ValueError(f"Invalid architecture: {arch}")
