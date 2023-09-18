#!/usr/bin/python
from optparse import OptionParser
import os
import sys
import numpy as np
import itertools
import re
from subprocess import call
import json
import llcommons
import shutil

import launch_exps

dat = {}

keyword_list = [
    # ('l1tex__t_sector_hit_rate.ratio', 'float'),
    # ('smsp__warp_issue_stalled_barrier_per_warp_active.pct', 'float'),
    # ('throughput', 'float'),
    # ('edgesNum', 'int'),
    # ('alwaysActiveNum', 'int'),
    # ('startActiveNum', 'int'),
    # ('filter-fetch-size', 'int'),
    # ('advance-fetch-size', 'int'),
    # ('filter-block-size', 'int'),
    # ('advance-fetch-size', 'int'),
]

def get_type_of_kw(kw):
    for k, t in keyword_list:
        if k == kw:
            return t

    return None

def get_token_after_symbol(st, kw, symbol):
   #print st,' ', k
    try:
        pos = st.index(kw)
        st = st[pos:]
        #print 'ppp = ', st
        tmp1 = st.split(symbol)
        return tmp1[1]
    except:
        return -1


def read_from_file(filepath, keyword_list, validation, doncu):
    file_data = {}
    # print(filepath) 

    for kw, typ in keyword_list:
        # print(kw, typ)
        #tmp = os.popen("cat %s | grep -w %s | awk '{print $3}'" % (filepath, kw) ).read()
        if doncu:
            try:
                tmp = os.popen("cat %s | grep -w %s " % (filepath, kw)).read()
                v = tmp.strip().split('\n')[0]
                lis = list(v.split(" "))
                if len(lis) == 0:
                    item = -1
                else:
                    item = lis[len(lis)-1]
            except:
                item = -1
        else:
            succ_string = ""
            if validation == True:
                succ_string = "Validation PASS"
            else:
                succ_string = "FINISHED"
            validate_output = os.popen("cat %s | grep \"%s\" " % (filepath, succ_string) ).read().strip()
            if  validate_output == "":
                timeout_string = "Timeout"
                timeout_output = os.popen("cat %s | grep \"%s\" " % (filepath, timeout_string) ).read().strip()
                if timeout_output == "":
                    item = -1 # validation failed
                else:
                    item = -2 # timeout
            else:
                # print("cat %s | grep -w %s " % (filepath, kw))
                tmp = os.popen("cat %s | grep -w %s " % (filepath, kw) ).read()
                #res_list.append(tmp.strip())
                v = tmp.strip().split('\n')[0]
                
                denghao = get_token_after_symbol(v, kw, '=')
                if denghao == -1:
                    maohao = get_token_after_symbol(v, kw, ':')
                    if maohao == -1:
                        item = -3 # parse failed
                    else:
                        item = maohao.strip().split(' ')[0]
                else:
                    item = denghao.strip().split(' ')[0]

        try:
            if isinstance(item, str):
                item = item.replace(",","")
            if get_type_of_kw(kw) == 'float':
                file_data[kw] = float(item)
            elif get_type_of_kw(kw) == 'int':
                file_data[kw] = int(item)
            else:
                file_data[kw] = item
        except:
            # print(item)
            file_data[kw] = -3 # parse failed

    return file_data


def select_data(dat, cfg, kw, app_list):
    res = []
    for it in app_list:
        res.append(dat[cfg][it][kw])
    
    return res


def print_to_file(out_fname, dat, keyword, app_list, cfg_list):
    print("Save to " + os.getcwd() + "/" + out_fname)
    f = open(out_fname, 'w')
    #head
    f.write("config")
    for app in app_list:
        f.write("," + app)
    f.write("\n")
    cfg_list.sort()
    for cfg in cfg_list:
        # print(cfg)
        #e = 'res_%d.txt' % cfg
        d = select_data(dat, cfg, keyword, app_list)
        f.write(cfg)

        for it in d:
            f.write("," + str(it))
        f.write("\n")

    f.close()


def calc_average(data1, exp_times, kw, app_list, cfg_list):
    res = llcommons.nested_dict()
    #data1[] [kw]
    #dat[exp_t][fname][app] = read_from_file(filepath, keyword_list)

    #[exp_t][fname][app]
    for exp in range(exp_times):
        for app in app_list:
            for cfg in cfg_list:
                res[cfg][app][kw] = 0

    for exp in range(exp_times):
        for app in app_list:
            for cfg in cfg_list:
                #print data1[exp][cfg][app][kw]
                res[cfg][app][kw] += data1[exp][cfg][app][kw]

    for cfg in cfg_list:
        for app in app_list:
            res[cfg][app][kw] /= exp_times

    return res


def calc_std(data1, num_exps, cfg, app, kw):
    #a = np.array([1,2,3,4])
    #print(np.std(a))
    arr = []
    for i in range(num_exps):
        arr.append(data1[i][cfg][app][kw])

    return np.std(np.array(arr))

def calc_std_table(data1, num_exps, kw, app_list, cfg_list):
    res_table = llcommons.nested_dict()

    for cfg in cfg_list:
        for app in app_list:
            res_table[cfg][app][kw] = calc_std(data1, num_exps, cfg, app, kw)

    return res_table


def calc_95_interval_table(std_data, num_exps, kw, app_list, cfg_list):
    res_table = llcommons.nested_dict()
    #std_data[cfg][app]
    #calc_ci(mean, std)
    for cfg in cfg_list:
        for app in app_list:
            res_table[cfg][app][kw] = llcommons.calc_ci(std_data[cfg][app][kw], num_exps)

    return res_table


dat_avg = {}

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="cfg_filename", help="config file for experiments", metavar="FILE")
    parser.add_option("-r", "--benchmark_root", dest="benchmark_root", help="root folder of benchmarks", metavar="FILE")
    parser.add_option("-k", "--keywords", dest="kw_file", help="specify the keyword list", metavar="FILE")
    parser.add_option("-b", "--benchmark_desc", dest="benchmark_desc_file", help="specify benchmark description file", metavar="FILE")
    parser.add_option("-n", "--nvprof", dest="nvprof_log", action="store_true", default=False, help="the log files are from nvprof?")
    parser.add_option("--validation", dest="validation", action="store_true", help="valiation with cpu version", default=False)
    parser.add_option("--csvdest", dest="csvdest", help="copy avgres_throughput.csv to csvdest", default="")
    parser.add_option("--ncu",  dest="doncu", action="store_true", help="profile the experiments via ncu", default=False)
    parser.add_option("--timeout-mins", dest="timeout_mins", help="time out minutes", default=120)

    (options, args) = parser.parse_args()
    #print options
    #print args

    cfg_file = options.cfg_filename
    print('read experiments configs from : ', cfg_file)

    cfg = launch_exps.create_config(cfg_file)

    if options.benchmark_root != None:
        cfg.set_benchmark_root_path(options.benchmark_root)

    if options.benchmark_desc_file != None:
        cfg.set_benchmark_desc(options.benchmark_desc_file)
        print('read apps from : ', options.benchmark_desc_file)


    kw_file = options.kw_file
    print('keyword_file =', kw_file)

    if kw_file != None:
        with open(kw_file) as f:
            kwdata = f.read()
        keyword_list = eval(kwdata)
    print("keyword_list = ", keyword_list)


    print('exp_times = ', cfg.get_exp_times())
    #cfg.set_do_exec(options.doexec)
    #cfg.set_do_nvprof(options.donvprof)
    #cfg.set_pbs_template(options.pbstempate)

    #print cfg.get_exp_times()
    #print cfg.get_excludes_apps()
    print('getapps = ', cfg.get_apps())

    cfg.generate_commands()
    # print(cfg.get_output_filenames())
    
    dat = llcommons.nested_dict()

    for exp_t in range(cfg.get_exp_times()):
        for app in cfg.get_apps():
            for fname in cfg.get_output_filenames():    
                filepath = ""
                if not options.nvprof_log:
                    filepath = '%s/%s_t%d.txt' % (app, fname, exp_t)
                else:
                    filepath = '%s/%s_t%d_nvprof.txt' % (app, fname, exp_t)
                
                if options.doncu:
                    filepath = '%s/%s_t%d_ncurep.txt' % (app, fname, exp_t)
                
                #print filepath
                enable_validation = options.validation and cfg.get_enable_validation_for_app(app)
                
                dat[exp_t][fname][app] = read_from_file(filepath, keyword_list, enable_validation, options.doncu)

    # for exp_t in range(cfg.get_exp_times()):
    #     for kw, w in keyword_list:
            # print_to_file('res_%s_t%d.txt' % (kw, exp_t), dat[exp_t], kw, cfg.get_apps(), cfg.get_output_filenames())


    # print(dat)

    std_table = {}
    for kw, w in keyword_list:
        dat_avg[kw] = calc_average(dat, cfg.get_exp_times(), kw, cfg.get_apps(), cfg.get_output_filenames())
        print_to_file('avgres_%s.csv' % kw, dat_avg[kw], kw, cfg.get_apps(), cfg.get_output_filenames())

    for kw, w in keyword_list:
        std_table[kw] = calc_std_table(dat, cfg.get_exp_times(), kw, cfg.get_apps(), cfg.get_output_filenames())
        print_to_file('std_%s.csv' % kw, std_table[kw], kw, cfg.get_apps(), cfg.get_output_filenames())

    ci_95_table = {}
    for kw, w in keyword_list:
        ci_95_table[kw] = calc_95_interval_table(std_table[kw], cfg.get_exp_times(), kw, cfg.get_apps(), cfg.get_output_filenames())
        print_to_file('ci95_%s.csv' % kw, ci_95_table[kw], kw, cfg.get_apps(), cfg.get_output_filenames())

    if options.csvdest:
        if options.doncu:
            print("copy", os.getcwd() + "/avgres_l1tex__t_sector_hit_rate.ratio.csv to", os.getcwd()+"/../../"+ "results/raw/ncu/l1cache/avgres_l1tex__t_sector_hit_rate" + options.csvdest)
            os.popen('cp ' + os.getcwd() + "/avgres_l1tex__t_sector_hit_rate.ratio.csv " + os.getcwd()+"/../../"+ "results/raw/ncu/l1cache/avgres_l1tex__t_sector_hit_rate" + options.csvdest) 
            print("copy", os.getcwd() + "/avgres_dram__bytes.sum.csv to", os.getcwd()+"/../../"+ "results/raw/ncu/memory/avgres_dram__bytes.sum" +options.csvdest)
            os.popen('cp ' + os.getcwd() + "/avgres_dram__bytes.sum.csv " + os.getcwd()+"/../../"+ "results/raw/ncu/memory/avgres_dram__bytes.sum" +options.csvdest) 
            print("copy", os.getcwd() + "/avgres_sass__inst_executed_global_loads.csv to", os.getcwd()+"/../../"+ "results/raw/ncu/memory/avgres_sass__inst_executed_global_loads" +options.csvdest)
            os.popen('cp ' + os.getcwd() + "/avgres_sass__inst_executed_global_loads.csv " + os.getcwd() +
                     "/../../" + "results/raw/ncu/memory/avgres_sass__inst_executed_global_loads" + options.csvdest)
            print("copy", os.getcwd() + "/avgres_sass__inst_executed_global_stores.csv to", os.getcwd() +
                  "/../../" + "results/raw/ncu/memory/avgres_sass__inst_executed_global_stores" + options.csvdest)
            os.popen('cp ' + os.getcwd() + "/avgres_sass__inst_executed_global_stores.csv " + os.getcwd() +
                     "/../../" + "results/raw/ncu/memory/avgres_sass__inst_executed_global_stores" + options.csvdest)

        else:
            print("copy", os.getcwd() + "/avgres_throughput.csv to", os.getcwd()+"/../../"+ options.csvdest)
            os.popen("mkdir -p " + os.path.dirname(os.getcwd()+"/../../"+ options.csvdest))
            os.popen('cp ' + os.getcwd() + "/avgres_throughput.csv " + os.getcwd()+"/../../"+ options.csvdest) 

