from .DFFormer_arch import DFFormer


def define_network(opt):
    opt = dict(opt)
    network_type = opt.pop("type")
    if network_type != "DFFormer":
        raise ValueError(f"Unsupported network type: {network_type}")
    return DFFormer(**opt)
