from __future__ import division
import numpy as np
import pandas as pd
import ast
from .params import enumerateParams, dictmerge
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import mpld3

mpld3.enable_notebook()

def parseLogs(logfile, kw="rawlogs:"):
    logs = {}
    dicts = []
    with open(logfile) as f:
        for i_line, line in enumerate(f):
            pos = line.find(kw)
            if pos >= 0:
                pos += len(kw) + 1

                pos_tensor = line.find("tensor(")
                while pos_tensor >= 0:
                    comma_end = line.find(",", pos_tensor)
                    tensor_end = line.find(")", pos_tensor)
                    line_new = line[:pos_tensor] + line[pos_tensor + len("tensor("):comma_end] + line[tensor_end+1:]
                    print("LINE", line)
                    print("NEW", line_new)
                    if len(line_new) >= len(line):
                        print("No line reduction")
                        break
                    line = line_new
                    pos_tensor = line.find("tensor(")

                try:
                    dic = ast.literal_eval(line[pos:])
                except Exception as e:
                    # print("Encountering weird patterns in logs")
                    # print("Line number %d" % i_line)
                    # print(line)
                    line = line.replace("nan,", "-1e8,")
                    line = line.replace("NaN", "-1e8")
                    line = line.replace("Infinity", "1e8")
                    try:
                        dic = ast.literal_eval(line[pos:])
                    except:
                        print("Unable to replace NaNs")
                        print(line[pos:])
                        continue

                for k in dic.keys():
                    if k not in logs:
                        logs[k] = []
                    logs[k].append(dic[k])
                dicts.append(dic)

    df = pd.DataFrame(dicts)

    return logs, df


def plotall(name, logs, var2range, metrics, outer_vars, inner_vars, color_by=None, pointer=True, ymin=None, ymax=None, loc=None, yscale={}):
    if not pointer:
        mpld3.disable_notebook()
    else:
        mpld3.enable_notebook()

    assert (color_by is None) or (color_by in inner_vars)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    inner_sets = enumerateParams(var2range, keys=inner_vars)

    for metric in metrics:
        for outer_set in enumerateParams(var2range, keys=outer_vars):
            for inner_set in inner_sets:
                # print outer_set, inner_set
                # print dictmerge(outer_set, inner_set)
                key = name.format(**dictmerge(outer_set, inner_set))

                # print(key, key in logs)
                # print(key, key in logs, len(logs[key]) >= 1, metric in logs[key])
                if key in logs and len(logs[key]) >= 1 and metric in logs[key]:
                    vals = logs[key][metric]
                    vals = vals[vals.notnull()].values
                    legend = ", ".join(["%s=%s" % (k, v) for k, v in inner_set.items()])

                    #if (ymin is not None) or (ymax is not None):
                    #    plt.ylim(ymin=ymin, ymax=ymax)

                    if ymin is not None:
                        vals = np.maximum(vals, ymin)
                    if ymax is not None:
                        vals = np.minimum(vals, ymax)
                    if color_by is None:
                        curve = plt.plot(vals, '.-', label=legend)
                        # curve = plt.plot(vals, 'k.-')
                    else:
                        curve = plt.plot(vals, '.-', color=colors[var2range[color_by].index(inner_set[color_by])], label=legend)
                        # curve = plt.plot(vals, 'k.-')

                    if pointer:
                        label_values = ['<div style="background-color: rgba(255, 255, 255, 0.8); font-size: 10px">%.4f (%d, %s)</div>' % (v, idx_v, legend) for idx_v, v in enumerate(vals)]
                        tooltips = mpld3.plugins.PointHTMLTooltip(curve[0], labels=label_values)
                        mpld3.plugins.connect(plt.gcf(), tooltips)

                if yscale.get(metric, None) == 'log':
                    plt.yscale('log')

            plt.title(metric + ", " + (", ".join(["%s=%s" % (k, v) for k, v in outer_set.items()])))
            if loc is None:
                plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
            else:
                plt.legend(loc=loc)
            plt.show()

def select_value(serie, metric, selection):
    vals = serie[metric]
    vals = vals[vals.notnull()]

    if selection == "max":
        return vals.max()
    elif selection == "min":
        return vals.min()
    elif selection == "last":
        try:
            return vals.values[-1]
        except:
            return -1
    else:
        raise NotImplementedError("'selection' must be max or last")


def plotmargin(name, logs, var2range, metric, color_by=None, pointer=True, selection="max", loc=None):
    if not pointer:
        mpld3.disable_notebook()
    else:
        mpld3.enable_notebook()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for var in var2range.keys():
        outer_sets = enumerateParams(var2range, keys=[var])
        xs = [outer_set[var] for outer_set in outer_sets]

        if len(xs) <= 1:
            continue
        for inner_set in enumerateParams(var2range, keys=[k for k in var2range.keys() if k != var]):
            indices, ys = [], []
            for outer_set in outer_sets:
                key = name.format(**dictmerge(outer_set, inner_set))
                if key in logs and len(logs[key]) >= 1 and metric in logs[key]:
                    ys.append(select_value(logs[key], metric, selection))
                    indices.append(xs.index(outer_set[var]))
            legend = ", ".join(["%s=%s" % (k, v) for k, v in inner_set.items()])
            if color_by is None or var == color_by:
                curve = plt.plot(indices, ys, ".-", label=legend)
            else:
                curve = plt.plot(indices, ys, ".-", color=colors[var2range[color_by].index(inner_set[color_by])], label=legend)

            plt.xticks(ticks=np.arange(len(xs)), labels=xs)
            if pointer:
                label_values = ['<div style="background-color: rgba(255, 255, 255, 0.8); font-size: 10px">%.4f (%s, %s=%s)</div>' % (ys[i], legend, var, str(xs[indices[i]])) for i in range(len(indices))]
                tooltips = mpld3.plugins.PointHTMLTooltip(curve[0], labels=label_values)
                mpld3.plugins.connect(plt.gcf(), tooltips)

        # if loc is None:
        #     plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
        # else:
        #     plt.legend(loc=loc)
        plt.title(var)
        plt.show()

def plotBoth(name, logs, var2range, metric1, metric2, color_by=None, pointer=True, selection="max", xlabel=None, ylabel=None):
    if not pointer:
        mpld3.disable_notebook()
    else:
        mpld3.enable_notebook()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    sets = enumerateParams(var2range)
    m1, m2, keys, c = [], [], [], []

    points_legend = {}
    for paramset in sets:
        key = name.format(**paramset)
        m1.append(select_value(logs[key], metric1, selection))
        m2.append(select_value(logs[key], metric2, selection))
        keys.append(key)
        if color_by is not None:
            val_idx = var2range[color_by].index(paramset[color_by])
            c.append(colors[val_idx])
            if val_idx not in points_legend:
                points_legend[val_idx] = Line2D([0], [0], color=colors[val_idx], lw=4)

    m1 = np.array(m1)
    m2 = np.array(m2)
    if color_by is None:
        curve = plt.scatter(m1, m2)
    else:
        curve = plt.scatter(m1, m2, c=c)

    if pointer:
        label_values = ['<div style="background-color: rgba(255, 255, 255, 0.8); font-size: 10px">(%.4f, %.2f), %s</div>' % (m1[i], m2[i], keys[i]) for i in range(len(keys))]
        tooltips = mpld3.plugins.PointHTMLTooltip(curve, labels=label_values)
        mpld3.plugins.connect(plt.gcf(), tooltips)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.legend(points_legend.values(), ["%s=%s" % (color_by, str(var2range[color_by][k])) for k in points_legend.keys()])
    plt.show()


def crossvalidate(name, logs, var2range, keys, metric_val, metric_test):
    outer_sets = enumerateParams(var2range, keys=keys)
    for outer_set in outer_sets:
        best_value = 0
        best_test = 0
        argbest = ""
        for inner_set in enumerateParams(var2range, keys=[k for k in var2range.keys() if k not in keys]):
            key = name.format(**dictmerge(outer_set, inner_set))
            if key in logs and len(logs[key]) >= 1 and metric_val in logs[key]:
                vals = logs[key][metric_val]
                vals = vals[vals.notnull()]
                this_best = vals.max()
                if this_best >= best_value:
                    best_value = this_best
                    vals = logs[key][metric_test]
                    vals = vals[vals.notnull()]
                    best_test = vals.max()
                    argbest = key

        outer_str = ", ".join(["%s=%s" % (k, v) for k, v in outer_set.items()])
        print("Best for %s: val=%.4f, test=%.4f (%s)" % (outer_str, best_value, best_test, argbest))


def getMetrics(name, logs, var2range, metrics, selection="last"):
    dicts = []
    for params_set in enumerateParams(var2range):
        key = name.format(**params_set)
        # if key in logs and len(logs[key]) >= 1 and metric in logs[key]:
        any_metric = (key in logs) and (len(logs[key]) >= 1) and any([metric in logs[key] for metric in metrics])
        if any_metric:
            metrics_dict = {
                metric: select_value(logs[key], metric, selection) if key in logs and len(logs[key]) >= 1 and metric in logs[key] else -1
                for metric in metrics
            }
            dicts.append(dictmerge(params_set, metrics_dict))

    df = pd.DataFrame(dicts)

    return df



def snippetToDic(s):
    splits = s.split("=")
    splits2 = []
    splits2.append(splits[0][1:]) # [1:] removes the leading '_'
    for i in range(1, len(splits) - 1):
        chunks = splits[i].split("_")
        splits2.append(chunks[0])
        splits2.append("_".join(chunks[1:]))
    splits2.append(splits[-1])

    assert len(splits2) == 2 * (len(splits) - 1)

    return {k: v for (k, v) in zip(splits2[::2], splits2[1::2])}

