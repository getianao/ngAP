#!/usr/bin/python
from optparse import OptionParser
import os
import sys
import numpy as np
import itertools
import re
import subprocess
from subprocess import call
import json
import llcommons
import shutil
import time


# exclude_apps = ["Hamming_N1000_l22_d5", "Hamming_N1000_l31_d10",
#                 "Levenshtein_l24d5", "Levenshtein_l37d10"]

# exclude_configs = ["o1-nonblocking-aas", "o1-nonblocking-unique"]

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BRIGHTRED = '\u001b[31;1m'
    
class Config:
    cfg = None
    output_filenames = []
    benchmark_rootpath = None
    benchmark_desc_file = None
    benchmark_desc_obj = None
    pbs_template = None
    do_exec = False
    do_nvprof = False
    do_ncu = False
    do_gputrace = False

    nvprof_metrics = None
    nvprof_events = None
    dryrun = None
    validation = None
    
    exclude_configs = []
    exclude_apps = []
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.output_filenames = []
        self.exp_start_id = 0
        self.nvprof_metrics = None
        self.nvprof_events  = None
        self.dryrun = None
        self.timeout_seconds = 7200

    def set_exclude(self):
        if "exclude_apps" in self.benchmark_desc_obj:
            self.exclude_apps = self.benchmark_desc_obj["exclude_apps"]
        if "exclude_configs" in self.cfg:
            self.exclude_configs = self.cfg["exclude_configs"]
        
        print("exclude_configs:", self.exclude_configs)
        print("exclude_apps:", self.exclude_apps)

    def set_dryrun(self, dr):
        self.dryrun = dr

    def set_do_gputrace(self, gputrace):
        self.do_gputrace = gputrace

    def set_exp_start_id(self, exp_start_id):
        self.exp_start_id = exp_start_id
        
    def set_timeout_seconds(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        print("timeout =", self.timeout_seconds/60, "minutes")

    def set_benchmark_root_path(self, benchmark_rootpath):
        self.benchmark_rootpath = benchmark_rootpath

    def set_benchmark_desc(self, b_desc):
        self.benchmark_desc_file = b_desc
        self.benchmark_desc_obj = eval(llcommons.read_file_to_string(self.benchmark_desc_file))

    def get_apps(self):

        if 'apps' in self.cfg and self.cfg["apps"] != "default":
            return self.cfg["apps"]

        if self.benchmark_rootpath != None:
            return get_apps_in_dir(self.benchmark_rootpath)

        elif self.benchmark_desc_file != None:
            self.cfg['apps'] = []

            for a in self.benchmark_desc_obj['apps']:
                self.cfg['apps'].append(a['name'])

            return self.cfg['apps']




    def get_excludes_apps(self):
        if not 'exclude_apps' in self.cfg:
            return []
        else:
            return self.cfg['exclude_apps']

    def get_exp_times(self):
        if not 'exp_times' in self.cfg:
            return 1
        else:
            return self.cfg['exp_times']

    def get_output_file_prefix(self):
        if not 'out_prefix' in self.cfg:
            return 'output'
        else:
            return self.cfg['out_prefix']

    def get_executable(self, cfg_name):
        assert(cfg_name in self.cfg['exp_parameters'])
        for tup in self.cfg['exp_parameters'][cfg_name]:
            if tup[0] == 'exec':
                return tup[1]
        return None


    def get_command_template(self, cfg_name, anml, input_file, app, isHS, quick_validation, isVASim = False):
        assert(cfg_name in self.cfg['exp_parameters'])
        #print('cfg_name = ', cfg_name)
        if isHS:
            cmd_str_template = '%s ' % (self.get_executable(cfg_name)) 
            for tup in self.cfg['exp_parameters'][cfg_name]:
                if not (len(tup) >= 3 and tup[2] == 'nocombination'):
                    cmd_str_template += ' -%s RUOZHIRUOZHI ' % tup[0]
            cmd_str_template += '%s %s ' % (anml, input_file) 
            cmd_str_template += "-v %s " % quick_validation 
        elif isVASim:
            cmd_str_template = '%s ' % (self.get_executable(cfg_name)) 
            for tup in self.cfg['exp_parameters'][cfg_name]:
                if not (len(tup) >= 3 and tup[2] == 'nocombination'):
                    cmd_str_template += ' -%s RUOZHIRUOZHI ' % tup[0]
            cmd_str_template += '%s %s ' % (anml, input_file) 
        else:
            cmd_str_template = '%s -a %s -i %s ' % (self.get_executable(cfg_name), anml, input_file)
            cmd_str_template += "--app-name=%s " % app
            for tup in self.cfg['exp_parameters'][cfg_name]:
                if not (len(tup) >= 3 and tup[2] == 'nocombination'):
                    cmd_str_template += ' --%s=RUOZHIRUOZHI ' % tup[0]
            cmd_str_template += "--quick-validation=%s " % quick_validation
        cmd_str_template = cmd_str_template.replace('RUOZHIRUOZHI', '%s')
        return cmd_str_template

    def get_input_suffix(self):
        if 'input_suffix' in self.cfg:
            return self.cfg['input_suffix']
        else:
            return '1MB'

    def get_input_file_path_for_app(self, app):
        if self.benchmark_rootpath != None:
            #print '??????????????????', self.benchmark_rootpath
            input_file_dir = os.path.join(self.benchmark_rootpath, app, 'inputs')
            input_file = llcommons.get_file_path(input_file_dir, self.get_input_suffix())
            #print input_file
            return input_file
        elif self.benchmark_desc_file != None:
            for a in self.benchmark_desc_obj['apps']:
                if a['name'] == app:
                    return os.path.join(self.benchmark_desc_obj['root'], a['input'])

            print('error, there is no such app in desc file', app)
            return None
        else:
            print('error, should provide benchmarks')
            
    def get_quick_validation_for_app(self, app):
        if self.benchmark_rootpath != None:
            return -1
        elif self.benchmark_desc_file != None:
            for a in self.benchmark_desc_obj['apps']:
                if a['name'] == app and "quick_validation" in a:
                    return a["quick_validation"]
            return -1
        return -1

    def get_enable_validation_for_app(self, app):
        if self.benchmark_rootpath != None:
            return True
        elif self.benchmark_desc_file != None:
            for a in self.benchmark_desc_obj['apps']:
                if a['name'] == app and "validation" in a:
                    return a["validation"]
            return True
        return True

    # def get_report_off_for_app(self, app):
    #     if self.benchmark_rootpath != None:
    #         return "false"
    #     elif self.benchmark_desc_file != None:
    #         for a in self.benchmark_desc_obj['apps']:
    #             if a['name'] == app and "report-off" in a:
    #                 return a["report-off"]
    #         return "false"
    #     return "false"

    def get_automata_file_path_for_app(self, app, isHS=False):
        if self.benchmark_rootpath != None:
            anml_file_dir  = os.path.join(self.benchmark_rootpath, app, 'anml')
            anml_file      = llcommons.get_anml(anml_file_dir)
            return anml_file
        elif self.benchmark_desc_file != None:
            for a in self.benchmark_desc_obj['apps']:
                if a['name'] == app:
                    if isHS:
                        if "hs" in a:
                            return os.path.join(self.benchmark_desc_obj['root'], a['hs'])
                        else:
                            return 'error, hs file ',  app
                    else:
                        return os.path.join(self.benchmark_desc_obj['root'], a['automata'])
            return 'error, anml file ',  app
        else:
            print('error, should provide benchmarks')


    def generate_command_for_app(self, cfg_name, app):
        assert(cfg_name in self.cfg['exp_parameters'])
        
        output_name = ""
        for tup in self.cfg['exp_parameters'][cfg_name]:
            if(len(tup) >= 2 and tup[0] == "output-name"):
                output_name = tup[1]
                break
        
        isHS = False
        isVASim = False
        if "hyperscan" in cfg_name:
            isHS = True
        elif "vasim" in cfg_name:
            isVASim = True
        list_of_list = []
    
        for tup in self.cfg['exp_parameters'][cfg_name]:
            if(len(tup) >= 2 and tup[0] == "app"):
                if(app in tup[1]):
                    break
                else:
                    return []
                
        for tup in self.cfg['exp_parameters'][cfg_name]:
            if(len(tup) >= 2 and tup[0] == "excludedApp"):
                if(app in tup[1]):
                    return []
                else:
                    break
   
        for tup in self.cfg['exp_parameters'][cfg_name]:
            if(len(tup) >= 5 and tup[3] == 'specific-app-option' and app in tup[4]):
                list_of_list.append(tup[2])
            elif not (len(tup) >= 3 and tup[2] == 'nocombination'):
                # normal parameter
                list_of_list.append(tup[1])

        input_file = self.get_input_file_path_for_app(app)
        anml_file  = self.get_automata_file_path_for_app(app, isHS)
        quick_validation = self.get_quick_validation_for_app(app)
        if "error" in anml_file:
            return []
        
        #print('input_file = ', input_file, ' anml = ', anml_file)

        cmd_template = self.get_command_template(cfg_name, anml_file, input_file, app, isHS, quick_validation, isVASim)
        # print(cmd_template)
        
        res = []
        for exp in range(self.exp_start_id, self.exp_start_id + self.get_exp_times()):
            for it in  itertools.product(*list_of_list):
                real_cmd = cmd_template % it
                
                output_filename_wo_exp_t = ""
                output_filename = ""
                output_filename_template = ""
                if(output_name != ""):
                    output_filename_template_wo_exp_t = output_name
                    output_filename_template = output_filename_template_wo_exp_t + "_t%s"
                    output_filename_wo_exp_t = output_filename_template_wo_exp_t
                    tp = []
                    # tp.extend(list(it))
                    tp.append(str(exp))
                    tp = tuple(tp)
                    #print output_filename_template
                    #print tp
                    output_filename = output_filename_template % tp
                    self.output_filenames.append(output_filename_wo_exp_t)
                else:
                    output_filename_template_wo_exp_t = self.get_output_file_prefix() + "_" + cfg_name + '_%s' * len(it) 
                    output_filename_template = output_filename_template_wo_exp_t + "_t%s"
                    output_filename_wo_exp_t = output_filename_template_wo_exp_t % it

                    tp = []
                    tp.extend(list(it))
                    tp.append(str(exp))
                    tp = tuple(tp)
                    #print output_filename_template
                    #print tp
                    output_filename = output_filename_template % tp
                    self.output_filenames.append(output_filename_wo_exp_t)
                
                res.append((real_cmd, output_filename))

        return res


    def generate_commands(self):
        assert('exp_parameters' in self.cfg)
        res = {}
        for app in self.get_apps():
            if not app in res:
                res[app] = []  

            for it in self.cfg['exp_parameters']:
                res[app].extend(self.generate_command_for_app(it, app))  

        #print(res)

        return res

    def get_output_filenames(self):
        return list(set(self.output_filenames))

    def create_pbs_tasks(self, app, command, cmdname, outputfile, pbsT):
        if self.pbs_template != None:
            m = {'APPNAMEAAAAAAAAAAAAAAAAAAAA': app, 'EXECCOMMANDDDDDDDDDDDDDDDDDDDDDDDDDDD' : command + " > " + outputfile}
            exec_pbs = llcommons.replace_string_based_on_map(pbsT, m)

            pbsfile_path = cmdname + ".pbs"
            pbsfile = open(pbsfile_path, 'w')
            pbsfile.write(exec_pbs)
            pbsfile.close()


    def launch_experiments(self):
        # print(self.cfg['exp_parameters'])
        commands = cfg.generate_commands()
        exp_parameters_keys = list(self.cfg['exp_parameters'].keys())
        print("exp_parameters = ", exp_parameters_keys)

        for app in commands:
            print(bcolors.OKBLUE + "app:", app + bcolors.ENDC)
            pbsT = None
            if self.pbs_template != None:
                pbsT = llcommons.read_file_to_string(self.pbs_template)

            os.chdir(app)
            # print(commands[app])
            for cmd, outfile in commands[app]:
                if self.do_exec:
                    exp_parameters_name = outfile
                    enable_validation = self.validation and self.get_enable_validation_for_app(app)
                    skip = False
                    for exclude_config in self.exclude_configs:
                        if exclude_config in exp_parameters_name:
                            skip = True
                            break
                    if skip or app in self.exclude_apps:
                        continue
                    isHS = False
                    isVASim = False
                    if "hyperscan" in exp_parameters_name:
                        isHS = True
                    elif "vasim" in exp_parameters_name:
                        isVASim = True
                    if isHS:
                        cmd = cmd + "--support"
                    elif isVASim:
                        cmd = cmd + "-t"
                    else:
                        if enable_validation:
                            cmd = cmd + "--validation=true"
                        else:
                            cmd = cmd + "--validation=false"
                    if self.do_ncu:
                        metric_list = [
                            "group:memory__first_level_cache_table",
                            "sass__inst_executed_global_loads",
                            "sass__inst_executed_global_stores",
                            # of bytes written to L2 for global atomics
                            "l1tex__m_l1tex2xbar_write_bytes_mem_global_op_atom",
                            # of bytes read from L2 into L1TEX M-Stage for global atomics
                            # "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_atom",
                            # of bytes requested for global atomics
                            "l1tex__t_bytes_pipe_lsu_mem_global_op_atom",
                            # of bytes requested that missed for global atomics
                            "l1tex__t_bytes_pipe_lsu_mem_global_op_atom_lookup_miss",
                            #    SM throughput assuming ideal load balancing across SMSPs
                            "sm__throughput.avg.pct_of_peak_sustained_active",
                            # proportion of warps per cycle, waiting for sibling warps at a CTA
                            "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
                            # proportion of warps per cycle, waiting on a memory barrier
                            "smsp__warp_issue_stalled_membar_per_warp_active.pct",
                            # cumulative # of warps in flight
                            "sm__warps_active.avg.pct_of_peak_sustained_active",
                            "sm__maximum_warps_per_active_cycle_pct",
                            # of cycles where local/global/shared writeback interface was active
                            "l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_active",
                            # of sector hits per sector
                            "l1tex__t_sector_hit_rate",  # L1/TEX hit rate
                            # proportion of L2 sector lookups that hit
                            "lts__t_sector_hit_rate", # L2 hit rate
                            # average # of active threads per instruction executed
                            "smsp__thread_inst_executed_per_inst_executed",
                            # of bytes accessed in DRAM
                            "dram__bytes",
                            # # of bytes read from DRAM                            
                            "dram__bytes_read",
                            # # of bytes written to DRAM
                            "dram__bytes_write",
                        ]

                        # cmd = "ncu --metrics " + ','.join(metric_list) + " -f -o " + outfile + " " + cmd
                        cmd = "ncu --set full -f -o " + outfile + " " + cmd
                        prog_output = outfile + "_ncu.txt"
                    else:
                        prog_output = outfile + ".txt"
                    print(app, outfile, 'exec command : ==== ', cmd, flush=True)
                    if not self.dryrun:
                        if self.pbs_template == None:
                            f_log = open(prog_output, 'w')
                            try:
                                tic = time.perf_counter()
                                call(cmd.split(), stdout=f_log, timeout=self.timeout_seconds)
                                toc = time.perf_counter()
                                succ_string = ""
                                if enable_validation == True:
                                    succ_string = "Validation PASS"
                                else:
                                    succ_string = "FINISHED"
                                validate_output = os.popen(
                                        "grep  \"%s\" ./%s" % (succ_string, prog_output)).read().strip()
                                if validate_output == "":
                                    print(bcolors.BRIGHTRED + exp_parameters_name,
                                        "Error: validation failed" + bcolors.ENDC, f"{toc - tic:0.1f} seconds", flush=True)
                                else:
                                    if not isHS and not isVASim:
                                        throughput = os.popen(
                                            "grep -w \"ngap elapsed time:\" ./%s |  awk \'{print $8}\'" % prog_output).read()[:-1]
                                        if throughput.strip() == "":
                                            throughput = os.popen(
                                            "grep -w \"throughput\" ./%s |  awk \'{print $3}\'" % prog_output).read()[:-1]
                                        if throughput.strip() != "":    
                                            print(bcolors.OKGREEN + exp_parameters_name, "throughput:",
                                                throughput, "MB/s, " + bcolors.ENDC, f"{toc - tic:0.1f} seconds", flush=True)
                                            if self.do_ncu:
                                                ncu_report = outfile + "_ncurep.txt"
                                                ncurep_cmd = "ncu --import " + outfile+".ncu-rep --print-units=base"
                                                print(ncurep_cmd)
                                                ncu_log = open(ncu_report, 'w') 
                                                try:
                                                    call(ncurep_cmd.split(), stdout=ncu_log, timeout=self.timeout_seconds)
                                                except Exception as e:
                                                    print ("get ncu report failed.")
                                                    print (str(e))      
                                        else:
                                            print(bcolors.WARNING + exp_parameters_name,
                                            "Error: get throughput failed" + bcolors.ENDC, f"{toc - tic:0.1f} seconds", flush=True)
                                    else:
                                        throughput = os.popen(
                                            "grep -w \"throughput\" ./%s |  awk \'{print $3}\'" % prog_output).read()[:-1]
                                        if throughput.strip() != "":    
                                            print(bcolors.OKGREEN + exp_parameters_name, "throughput:",
                                                throughput, "MB/s, " + bcolors.ENDC, f"{toc - tic:0.1f} seconds", flush=True) 
                                        else:
                                            print(bcolors.WARNING + exp_parameters_name,
                                            "Error: get throughput failed" + bcolors.ENDC, f"{toc - tic:0.1f} seconds", flush=True)           
                                
                            except subprocess.TimeoutExpired:
                                print(bcolors.WARNING + exp_parameters_name,
                                      "Error: timeout" + bcolors.ENDC, f"{self.timeout_seconds:0.1f} seconds", flush=True)
                                f_log.close()
                                f_log_a = open(prog_output, "a")
                                f_log_a.write("\n" + "Timeout " + f"{self.timeout_seconds:0.1f} seconds")
                                f_log_a.close()
                        else:
                            self.create_pbs_tasks(app, cmd, outfile, prog_output, pbsT)
                            call(['qsub', outfile + '.pbs'])

                # nv_mode1_log = outfile + "_nvprof.txt"
                # if nv_mode1_log.find('_t%d_' % self.exp_start_id) != -1:
                #     if self.do_nvprof:
                #         nv_mode1 = "nvprof  --metrics %s --csv --events %s --log-file  %s_nvprof.csv " % (self.nvprof_metrics, self.nvprof_events, outfile)
                #         nv_cmd = nv_mode1 + cmd
                #         nv_mode1_log = outfile + "_nvprof.txt"
                #         print('NVPROF COMMAND 1: ', nv_cmd)
                #         if not self.dryrun:
                #             if self.pbs_template == None:
                #                 nv_mode1_log_file = open(nv_mode1_log, "w")
                #                 call(nv_cmd.split(), stdout=nv_mode1_log_file)
                #             else:
                #                 self.create_pbs_tasks(app, nv_cmd, 'nvprof_' + outfile , nv_mode1_log, pbsT)
                #                 call(['qsub', 'nvprof_' + outfile + '.pbs'])
                    
                #     if self.do_gputrace:
                #         nv_gputrace = "nvprof  --csv --log-file  %s_gputrace.csv --print-gpu-trace " % outfile
                #         nv_gputrace_cmd = nv_gputrace + cmd
                #         nv_gputrace_log = outfile + "_gputrace.txt"
                #         print('NVPROF COMMAND 2: ', nv_gputrace_cmd)
                #         if not self.dryrun:
                #             if self.pbs_template == None:
                #                 nv_gputrace_log_file = open(nv_gputrace_log, "w")
                #                 call(nv_gputrace_cmd.split(), stdout=nv_gputrace_log_file)
                #             else:
                #                 self.create_pbs_tasks(app, nv_gputrace_cmd, 'nvprofgputrace_' + outfile , nv_gputrace_log, pbsT)
                #                 call(['qsub', 'nvprofgputrace_' + outfile + '.pbs'])
                    
            os.chdir('..')

    def set_do_exec(self, doexec):
        self.do_exec = doexec

    def set_do_nvprof(self, donvprof):
        self.do_nvprof = donvprof

    def set_do_ncu(self, ncu):
        self.do_ncu = ncu

    def set_pbs_template(self, pbstemplate):
        self.pbs_template = pbstemplate

    def set_nvprof_metrics(self, m):
        self.nvprof_metrics = m

    def set_nvprof_events(self, m):
        self.nvprof_events = m

    def set_validation(self, m):
        self.validation = m

    def __repr__(self):
        return self.cfg



def create_config(config_file):
    with open(config_file) as f:
        data = f.read()

    return Config(eval(data))


def get_apps_in_dir(rootpath):
    return llcommons.get_layer1_folders(rootpath)


def clean_folders(dir_list, path="."):
    print ("clean --- ", dir_list)
    for d in dir_list:
        try:
            #if clean_old_folders:
            print ("remove", os.path.join(path, d))
            shutil.rmtree(os.path.join(path, d))
        except Exception as e:
            #print "error"
            print (str(e))
            #pass

def create_folders(dir_list, path='.'):
    for d in dir_list:
        llcommons.create_dirs_on_path(os.path.join(path, d, 'aaa'))


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="cfg_filename", help="config file for experiments", metavar="FILE")

    parser.add_option("-r", "--benchmark_root", dest="benchmark_root", help="root folder of benchmarks", metavar="FILE")
    parser.add_option("-b", "--benchmark_desc", dest="benchmark_desc_file", help="specify benchmark description file", metavar="FILE")
    
    parser.add_option("-e", action="store_true", dest="doexec", help="execute the experiments")
    parser.add_option("-p", action="store_true", dest="donvprof", help="profile the experiments")
    parser.add_option("--gputrace", dest="gputrace", action="store_true", help="nvprof gputrace", default=False)
    parser.add_option("-t", "--usepbs", dest="pbstempate", help="template of pbs", metavar="FILE")
    
    parser.add_option("-c", "--clean", action="store_true", dest="cleanoldfolders", default=False, help="clean experiment folders based on current exp config")
    
    parser.add_option("--exp-start-id", dest="exp_start_id", help="the number (index) of exp when start the experiments")
    parser.add_option("--timeout-mins", dest="timeout_mins", help="time out minutes", default=120)

    parser.add_option("--metrics", dest="metrics", help="metrics used in nvprof",  default="all")
    parser.add_option("--events", dest="events", help="events used in nvprof", default="all")
    parser.add_option("--dryrun", action="store_true", dest="dryrun", help="only show commands without running them", default=False)
    parser.add_option("--validation", dest="validation", action="store_true", help="valiation with cpu version", default=False)
    parser.add_option("--csvdest", dest="csvdest", help="copy avgres_throughput.csv to csvdest", default="")
    parser.add_option("--ncu",  dest="doncu", action="store_true", help="profile the experiments via ncu", default=False)
    parser.add_option("-k", "--keywords", dest="kw_file", help="specify the keyword list (do nothing)", metavar="FILE")


    (options, args) = parser.parse_args()
    
    cfg_file = options.cfg_filename
    print ('read experiments configs from : ', cfg_file)
    
    cfg = create_config(cfg_file)
    if options.exp_start_id != None:
        cfg.set_exp_start_id(int(options.exp_start_id))

    if options.benchmark_root != None:
        cfg.set_benchmark_root_path(options.benchmark_root)

    if options.benchmark_desc_file != None:
        cfg.set_benchmark_desc(options.benchmark_desc_file)
        print('read apps from : ', options.benchmark_desc_file)

    cfg.set_timeout_seconds(int(options.timeout_mins)* 60)
    cfg.set_do_exec(options.doexec)
    cfg.set_do_nvprof(options.donvprof)
    cfg.set_do_ncu(options.doncu)
    cfg.set_pbs_template(options.pbstempate)
    cfg.set_dryrun(options.dryrun)
    cfg.set_nvprof_events(options.events)
    cfg.set_nvprof_metrics(options.metrics)
    cfg.set_do_gputrace(options.gputrace)
    cfg.set_validation(options.validation)

    #print cfg.get_exp_times()
    #print cfg.get_excludes_apps()
    cfg.set_exclude()
    print('getapps = ', cfg.get_apps())

    if options.cleanoldfolders:
        #clean_folders(cfg.get_apps(), '.')
        if not options.dryrun:
            os.system('rm -Rf -- */')

    create_folders(cfg.get_apps(), '.')
    
    cfg.launch_experiments()
