#!/usr/bin/python
import os, errno
import collections
import scipy
import scipy.stats
import pandas as pd
import argparse
import math

# begin from Prof. Sree  -----------------------------
def critlevel(n, level_perc):
    import scipy.stats
    # not the same alpha as in the eqns ...
    alpha = level_perc / 100.0

    if n > 32:
        return scipy.stats.norm.interval(alpha)[1]
    else:
        return scipy.stats.t.interval(alpha, n - 1)[1]

def calc_ci(stdev, n, level_perc=95):
    t1 = critlevel(n, level_perc)
    se = stdev / math.sqrt(n)
    zt = t1*se
    return zt
# end  ------------------------------------------


nested_dict = lambda: collections.defaultdict(nested_dict)

def get_layer1_folders(path):
    d = path
    print(path)
    return filter(lambda x: os.path.isdir(os.path.join(d, x)), os.listdir(d))


def get_layer1_files(mypath):
    print(mypath)
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    return onlyfiles


def create_dirs_on_path(filepath):
    if not os.path.exists(os.path.dirname(filepath)):
        try:
            os.makedirs(os.path.dirname(filepath))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def replace_string_based_on_map(ss, mp):
    ss1 = ss
    for kw in mp:
        ss1 = ss1.replace(kw, mp[kw])

    return ss1


def read_file_to_string(filepath):
    #print os.getcwd()
    with open(filepath, 'r') as myfile:
        data=myfile.read()

    return data


def get1Minput(path):
    for subdir, dirs, files in os.walk(path):
        #print files
        for ff in files:
            if ff.find('1MB') != -1:
                return ff


def get_anml(path):
    for subdir, dirs, files in os.walk(path):
        #print files
        for ff in files:
            if ff.endswith('.anml'):
                return os.path.abspath(os.path.join(path, ff))

def get_hs(path):
    for subdir, dirs, files in os.walk(path):
        #print files
        for ff in files:
            if ff.endswith('.hs'):
                return os.path.abspath(os.path.join(path, ff))


def get_file_path(path, suffix):
    files = get_layer1_files(path)

    res = []
    for f in files:
        assert(os.path.isfile(os.path.join(path, f)))

        filename_wo_ext = os.path.splitext(f)[0]
        if filename_wo_ext.endswith(suffix):
            res.append(os.path.abspath(os.path.join(path, f)))

    assert(len(res) == 1)

    return res[0]




def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h



if __name__ == '__main__':
    # just a few tests.
    print(get_layer1_folders('../benchmarks'))
    print(get_layer1_files('../benchmarks/Brill/inputs'))
    for app in get_layer1_folders('../benchmarks'):
        print(get_file_path('../benchmarks/%s/inputs' % app, '1MB'))
    
    