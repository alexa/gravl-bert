# Original Copyright (c) 2021 Matthias Fey, Jiaxuan You
# Modifications Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import sys
import os.path as osp
from getpass import getuser
from tempfile import NamedTemporaryFile as TempFile, gettempdir
from importlib.util import module_from_spec, spec_from_file_location



def class_from_module_repr(cls_name, module_repr):
    path = osp.join(gettempdir(), f'{getuser()}_pyg')
    os.makedirs(path)
    with TempFile(mode='w+', suffix='.py', delete=False, dir=path) as f:
        f.write(module_repr)
    spec = spec_from_file_location(cls_name, f.name)
    mod = module_from_spec(spec)
    sys.modules[cls_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)