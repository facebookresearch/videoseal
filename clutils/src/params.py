# import random
from collections import OrderedDict

def enumerateParams(params, keys: list=None):
    """
    Enumerates all configurations of parameters in a sweep
    ex1:
    params = {'a': [0,1], 'b': [17, 19]}
    returns [{'a': 0, 'b': 17},
             {'a': 0, 'b': 19},
             {'a': 1, 'b': 17},
             {'a': 1, 'b': 19}]

    ex2:
    params = {'a': {
                '0': {'b': 17},
                '1'" {'b': 19}
               }
              }
    returns [{'a': 0, 'b': 17},
             {'a': 1, 'b': 19}]

    If a parameter takes only 1 value it can be specified as either a
    list of length 1, or just the value
    """
    if keys is None:
        keys = list(params.keys())

    param_name = keys[0]
    local_params_enumerated = []
    # Cf. ex.2 above: val corresponds to '0' or '1' for 'a' (and val_params resp. corresponds to {'b': 17} and {'b': 19})
    if type(params[param_name]) in [dict, OrderedDict]:
        for param_value, params_sub in params[param_name].items():
            assert type(params_sub) in [dict, OrderedDict], "Subsweeps should have dicts"

            params_sub_enumerated = enumerateParams(params_sub)
            local_params_enumerated += [dictmerge(r, {param_name: param_value}) for r in params_sub_enumerated]
    else:
        if type(params[param_name]) != list:
            params[param_name] = [params[param_name]]
        local_params_enumerated = [{param_name: v} for v in params[param_name]]

    if len(keys) == 1:
        return local_params_enumerated
    else:
        recParams = enumerateParams(params, keys[1:])
        return [dictmerge(r, local_params_assignment) for r in recParams for local_params_assignment in local_params_enumerated]


def dictmerge(x, y):
    z = x.copy()
    z.update(y)

    return z


def generateExt(params, original, to_index=[]):
    """ Generate the file name extension that corresponds to the particular
    set of parameters. Parameters that can only take 1 value are not included """
    s = ""
    for k, v in params.items():
        if type(v) == list:
            v = tuple(v)
        if type(original[k]) == list and len(original[k]) >= 2:
            if k in to_index:
                val = str(original[k].index(v))
            else:
                val = str(v)
            s += "_" + k + "=" + val

    if s == "":
        s = "expe"

    return s
