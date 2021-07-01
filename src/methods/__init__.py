import importlib


def lazy_build_network(module, pass_predict_type=True, **kwargs):
    def build_network(arch_args, predict_type):
        extra_args = kwargs.copy()
        if pass_predict_type:
            extra_args.update({"predict_type": predict_type})
        mod = importlib.import_module(f".{module}", __name__)
        return mod.build_network(arch_args, **extra_args)
    return build_network


def build_knn(arch_args, predict_type):
    neighbors = importlib.import_module("sklearn.neighbors")
    return neighbors.KNeighborsRegressor(n_neighbors=1)


NETWORK_TYPES = {
    "mlp": lazy_build_network("mlp"),
    "cnn": lazy_build_network("cnn"),
    "unet": lazy_build_network("unet"),
    "nn-kernel": lazy_build_network("nn_kernel"),
    "knn": build_knn,
}


def get_network_type(arch):
    if arch in {"knn-regressor", "knn-predictor",
                "knn-regressor-oneshot", "knn-predictor-oneshot"}:
        return "knn", None
    for suffix in ["step", "deriv"]:
        dash_suffix = f"-{suffix}"
        if arch.endswith(dash_suffix):
            return arch[:-len(dash_suffix)], suffix
    raise ValueError(f"Invalid architecture: {arch}")


def build_network(net_args):
    arch = net_args["arch"]
    arch_key, predict_type = get_network_type(arch)
    arch_args = net_args["arch_args"]
    try:
        net_func = NETWORK_TYPES[arch_key]
    except KeyError as exc:
        raise ValueError(f"Invalid architecture: {arch}") from exc
    return net_func(arch_args, predict_type=predict_type)
