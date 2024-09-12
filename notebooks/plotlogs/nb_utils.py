
import json
import numpy as np

from skimage.metrics import peak_signal_noise_ratio

import matplotlib as mpl
import matplotlib.pyplot as plt

# put ticks inside the plot
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# put ticks on the up and right side also
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
# grid in very light grey
plt.rcParams['grid.alpha'] = 0.2
# put a grid by default
plt.rcParams['axes.grid'] = True

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

def read_params(params_file):
    with open(params_file, "r") as f:
        l = []
        for line in f:
            l.append(line.rstrip())
        d = {}
        for ii, line in enumerate(l):
            params_line = line.split("--")
            path = params_line.pop(0)
            path = path.split('	')[0]
            params = {}
            for jj, param in enumerate(params_line):
                param = param.split(" ")
                if param[-1]=='':
                    param = param[:-1]
                for ii in range(len(param)):
                    if isfloat(param[ii]): param[ii] = float(param[ii])
                    elif isint(param[ii]): param[ii] = int(param[ii])
                params[param[0]] = param[1:] if len(param)>2 else param[1]
            d[path] = params
        return d

def parse_logs(path, keyword):
    with open(path, "r") as f:
        l = []
        for line in f:
            l.append(line.rstrip())
        d = {}
        for ii, line in enumerate(l):
            if keyword in line:
                d[ii] = json.loads(line)
        return d

def filter_paths(params, filters):
    paths = list(params.keys())
    paths_copy = list(params.keys())
    iis = []
    for ii, path in enumerate(paths_copy):
        param = params[path]
        for k_filter, v_filter in filters.items():
            if k_filter in param:
                if param[k_filter] not in v_filter:
                    paths.remove(path)
                    break
    return paths

def sweepable_params(params):
    sweepable = {}
    for path, param in params.items():
        for k, v in param.items():
            if k not in sweepable:
                sweepable[k] = []
            if v not in sweepable[k]:
                sweepable[k].append(v)
    sweepable = {k: v for k, v in sweepable.items() if len(v)>1}
    sweepable = {k: v for k, v in sweepable.items() if 'master_port' not in k.lower()}
    return sweepable

def visu_diff(img_ori, img_comp, title=None, figsize=(20,30), crop=None, hori=True):
    mpl.rcParams['axes.grid'] = False

    plt.figure(figsize=figsize)
    
    img_ori_ar = np.asarray(img_ori)
    img_comp_ar = np.asarray(img_comp)

    # Crop the images if crop parameter is provided
    if crop is not None:
        if isinstance(crop, int):
            img_ori_ar = img_ori_ar[:crop, :crop]
            img_comp_ar = img_comp_ar[:crop, :crop]
        elif isinstance(crop, tuple):
            img_ori_ar = img_ori_ar[:crop[0], :crop[1]]
            img_comp_ar = img_comp_ar[:crop[0], :crop[1]]

    if hori:
        plt.subplot(1, 3, 1)    
    else:
        plt.subplot(3, 1, 1)
    plt.imshow(img_ori_ar)
    plt.title('Image 1')

    if hori:
        plt.subplot(1, 3, 2)    
    else:
        plt.subplot(3, 1, 2)
    plt.imshow(img_comp_ar)
    plt.title('Image 2')

    diff = img_comp_ar.astype(int) - img_ori_ar.astype(int)
    psnr = peak_signal_noise_ratio(img_ori_ar, img_comp_ar)
    print(f'PSNR: {psnr}')
    print(f'Linf: {np.max(np.abs(diff))}')
    
    # normalize diff
    diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
    diff = np.abs(diff - 0.5)
    # diff = np.linalg.norm(diff, ord=1, axis=2)
    # diff = diff * (1.0/diff.max())
    # diff = np.abs(diff)
    # diff = diff * (5/255)
    if hori:
        plt.subplot(1, 3, 3)    
    else:
        plt.subplot(3, 1, 3)
    # print(diff.shape)
    # diff[..., 1] = 0
    # diff[..., 0] = 0
    plt.imshow(diff)
    plt.title('Difference')

    title += f' - PSNR: {psnr:.2f}'
    if title is not None:
        plt.suptitle(title, fontsize=10)

    plt.tight_layout()
    plt.show()

    mpl.rcParams['axes.grid'] = True

    return diff

def remove_outliers(df, measure):
    Q1 = df[measure].quantile(0.25)
    Q3 = df[measure].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[measure] >= lower_bound) & (df[measure] <= upper_bound)]
