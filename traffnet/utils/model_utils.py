import ast
import contextlib

import torch
from torch import nn
from traffnet.moduels import (
    PathInputEmbedding,
    InnerPathModel,
    RouteSelectModel,
    TemporalGRU,
    PathConcatenation,
)
from traffnet.utils.data_utils import getSeqMaxSize


def parse_model(logger, d, config):
    path_seq_maxsize = getSeqMaxSize(file_path=config["dataset"]["data_path"])
    layers, save = [], []
    for i, (f, m, args) in enumerate(d):  # from, module, args
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        if m in (InnerPathModel, RouteSelectModel, PathConcatenation):
            args.append(path_seq_maxsize)
        m_ = m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        logger.info(f"[{i:>3}{str(f):>20}{m.np:10.0f}  {t:<45}{str(args):<30}]")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
    return nn.Sequential(*layers), sorted(save)
