import yaml
import os.path as osp


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def process_func(args):
    self, fns, idx_file = args
    return self.process_multiple_files(fns, idx_file)


def get_split_files(config_path, split, sigdir, bkgdir):
    with open(config_path, "r") as fi:
        data = yaml.safe_load(fi)
        paths = data[split]["paths"]
    paths_fin = []
    for path in paths:
        p = path.replace(osp.dirname(path), sigdir if "ZH" in path else bkgdir)
        paths_fin.append(p)
    return paths_fin
