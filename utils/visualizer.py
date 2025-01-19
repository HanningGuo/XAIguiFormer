import matplotlib.pyplot as plt
from bytecode import Bytecode, Instr
'''
An awesome visual tool to extract intermediate features with three lines statement, such as attention map
https://github.com/luo3300612/Visualizer
'''


class get_local(object):
    cache = {}
    is_activate = False

    def __init__(self, varname):
        self.varname = varname

    def __call__(self, func):
        if not type(self).is_activate:
            return func

        type(self).cache[func.__qualname__] = []
        c = Bytecode.from_code(func.__code__)
        extra_code = [
                         Instr('STORE_FAST', '_res'),
                         Instr('LOAD_FAST', self.varname),
                         Instr('STORE_FAST', '_value'),
                         Instr('LOAD_FAST', '_res'),
                         Instr('LOAD_FAST', '_value'),
                         Instr('BUILD_TUPLE', 2),
                         Instr('STORE_FAST', '_result_tuple'),
                         Instr('LOAD_FAST', '_result_tuple'),
                     ]
        c[-1:-1] = extra_code
        func.__code__ = c.to_code()

        def wrapper(*args, **kwargs):
            res, values = func(*args, **kwargs)
            type(self).cache[func.__qualname__].append(values.detach().cpu().numpy())
            return res
        return wrapper

    @classmethod
    def clear(cls):
        for key in cls.cache.keys():
            cls.cache[key] = []

    @classmethod
    def activate(cls):
        cls.is_activate = True


def plot_attention_maps(attention_maps):
    num_figures = len(attention_maps[-1].mean(axis=0))
    nrows = 2
    ncols = int(num_figures / nrows + 0.5)
    attention_figures, attention_axs = plt.subplots(nrows, ncols)
    for idx_ax, attention_map in enumerate(attention_maps[-1].mean(axis=0)):
        image = attention_axs[int(idx_ax / ncols)][idx_ax % ncols].imshow(attention_map)
    plt.colorbar(image, ax=attention_axs, orientation='horizontal')
    return attention_figures
