from functools import partial
from collections import OrderedDict
from math import ceil
from os.path import join
import argparse
import datetime
import json
import numpy as np
import os
import shutil
import subprocess
import tempfile
import time
import pandas as pd

from src.date import GMT1
from src.functions import cmdPre, loadlist, savelist
from src.functions import run_command, getuser, replaceMacros, bool_flag, linearize_params
from src.params import enumerateParams, generateExt

basename = os.path.basename

IS_FAIR_CLUSTER = os.path.exists("/checkpoint")
IS_AWS_CLUSTER = os.path.exists("/checkpoints")
IS_NEW_CLUSTER = IS_FAIR_CLUSTER or IS_AWS_CLUSTER

def step_number(s):
    if np.mean(s == 'PENDING') == 1.0:
        return 0
    elif np.any(s == 'RUNNING'):
        return 1.5
    elif np.any(s == 'FAILED'):
        return 1
    elif np.all(s == 'FAILED'):
        return 2
    elif np.all(s == 'FINISHED'):
        return 3
    else:
        return 4

def update_status(expdir, s):
    #snippet, status = line.strip().split('\t')
    failed = check_failed(join(expdir, 'logs', s.snippet + '.stderr'))
    finished = check_finished(join(expdir, 'logs', s.snippet + '.stdout'))
    started = check_started(join(expdir, 'logs', s.snippet + '.stdout'))

    if finished:
        return "FINISHED"
    elif failed:
        return "FAILED"
    elif started:
        return "RUNNING"
    else:
        return "PENDING"


def trash_expe(expe, trash_dir):
    """
    Moves experiment folder to trash_dir
    """
    dirs = [x for x in expe.split("/") if x != ""]
    base_level = len([x for x in ckpt_default().split("/") if x != ""])

    assert os.path.exists(expe), "Experiment does not exist"
    assert expe.startswith(ckpt_default()), "Directory should start with default checkpoint"
    assert len(dirs) - base_level == 1, "Experiment should be 1 levels below main checkpoint directory"

    dirs.append(datetime.datetime.now(tz=GMT1()).strftime('%Y%m%d_%H%M%S'))
    dst = join(trash_dir, "_".join(dirs[base_level:]))
    print("Moving %s to %s" % (expe, dst))
    shutil.move(expe, dst)

def check_started(std_path):
    cmd = f"grep -c -ni -P 'Beginning program' {std_path}  || true" # Hack to have exit code of 0
    output = run_command(cmd)
    return int(output) != 0

def check_failed(std_path):
    cmd = f"grep -c -ni -P 'srun(?!.* step creation temporarily disabled, retrying)(?!.* Step created for job)' {std_path}  || true" # Hack to have exit code of 0
    output = run_command(cmd)
    try:
        return int(output) != 0
    except:
        return False

def check_finished(std_path):
    cmd = f"grep -c -ni -P 'JOB_FINISHED' {std_path}  || true" # Hack to have exit code of 0
    output = run_command(cmd)
    return int(output) != 0

def ckpt_default():
    year = datetime.date.today().year
    user = getuser()
    if IS_FAIR_CLUSTER:
        return f"/checkpoint/{user}/{year}_logs/"
    elif IS_AWS_CLUSTER:
        return f"/checkpoints/{user}/{year}_logs/"
    else:
        return f"/checkpoint/{user}/{year}_logs/"

def jobname_default():
    year = datetime.date.today().year
    user = getuser()
    if IS_FAIR_CLUSTER:
        return f"/checkpoint/{user}/{year}_logs/jobnames.txt"
    elif IS_AWS_CLUSTER:
        return f"/checkpoints/{user}/{year}_logs/jobnames.txt"
    else:
        return f"/checkpoint/{user}/{year}_logs/jobnames.txt"

def stool_stress(partition):
    """
    Number of used CPU and GPU per user at `parititon`.
    """
    STRING = ("squeue " + partition + " --format \"%t %u %b\" "
              "| grep \"^R\" "
              "| grep -v \"null\" "
              "| cut -b3- "
              "| sed -e \"s/gpu://g\" "
              "| sed -e \"s/volta://g\" "
              "| awk '{arr[$1]+=$2} END {for (i in arr) {print i,arr[i]}}' "
              "| sort -n -k2 "
              "| awk '{printf \"%5s %s\\n\", $2, $1}'")
    subprocess.call(STRING, shell=True)

    STRING = ("squeue " + partition + " --format \"%t %b\" "
              "| grep gpu "
              "| grep R "
              "| cut -d\":\" -f2 "
              "| paste -sd+ "
              "| bc"
              "| awk '{printf \"%s%s\\n\", \"==== Total GPU: \", $1}'")
    subprocess.call(STRING, shell=True)

    print()

    STRING = ("squeue " + partition + " --format \"%t %u %C\" "
              "| grep \"^R\" "
              "| grep -v \"null\" "
              "| cut -b3- "
              "| sed -e \"s/gpu://g\" "
              "| sed -e \"s/volta://g\" "
              "| awk '{arr[$1]+=$2} END {for (i in arr) {print i,arr[i]}}' "
              "| sort -n -k2 "
              "| awk '{printf \"%5s %s\\n\", $2, $1}'")
    subprocess.call(STRING, shell=True)

    STRING = ("squeue " + partition + " --format \"%t %C\" "
              "| grep \"^R\" "
              "| cut -b3- "
              "| paste -sd+ "
              "| bc"
              "| awk '{printf \"%s%s\\n\", \"==== Total CPU: \", $1}'")
    subprocess.call(STRING, shell=True)

    print()

    STRING = ("echo You have "
              "$(squeue -u $(whoami) --format %t | grep R | wc -l) "
              "running and "
              "$(squeue -u $(whoami) --format %t | grep PD | wc -l) "
              "pending jobs.")

    subprocess.call(STRING, shell=True)

    return None

if IS_NEW_CLUSTER:
    blacklist = []
    sbatch = "sbatch "
    if len(blacklist) >= 1:
        sbatch += "--exclude "
        sbatch += ",".join(blacklist)
        sbatch += " "
else:
    sbatch = "/usr/local/chronos/bin/crun --hostgroup fblearner_ash_cpuram_default "


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt-root", type=str, default=ckpt_default())
parser.add_argument("--jobnames", type=str, default=jobname_default())
parser.add_argument("--trash-dir", type=str, default=join(ckpt_default(), "trash"))
parser.add_argument("--launch-parallel", action='store_true', dest='launch_parallel')
parser.set_defaults(launch_parallel=False)
subparsers = parser.add_subparsers(dest='command')

parser_count = subparsers.add_parser("count")
parser_count.add_argument("grid", type=str)

parser_list = subparsers.add_parser("list")
parser_list.add_argument("group", type=str)

parser_remove = subparsers.add_parser("remove")
parser_remove.add_argument("expe", type=str)

parser_sweep = subparsers.add_parser('sweep')
parser_sweep.add_argument('grid', type=str)
parser_sweep.add_argument("--no-launch", action='store_false', dest='launch')
parser_sweep.add_argument("--sample", type=int, default=-1)
parser_sweep.add_argument("--numeric", action='store_true', dest='numeric')
parser_sweep.add_argument("--array", type=bool_flag, default=True)
parser_sweep.add_argument("--pooling", type=int, default=1)
parser_sweep.set_defaults(launch=True, numeric=False)

parser_status = subparsers.add_parser("status")

parser_relaunch = subparsers.add_parser("relaunch")
parser_relaunch.add_argument("jobname", type=str)

parser_cancel = subparsers.add_parser("cancel")
parser_cancel.add_argument("jobname", type=str)

parser_check = subparsers.add_parser("check")

parser_stress = subparsers.add_parser('stress')
parser_stress.add_argument("--partition", type=str, default="")

parser_nb = subparsers.add_parser('notebook')
parser_nb.add_argument("json_path", type=str)
parser_nb.add_argument("nb_path", type=str)

args = parser.parse_args()
if not IS_NEW_CLUSTER:
    args.launch_parallel = True
# print(args)

if args.command == 'sweep':
    assert os.path.exists(args.grid), "Config file %s does not exist. Are you in the right repository ?" % args.grid
    config = json.load(open(args.grid), object_pairs_hook=OrderedDict)
    if "pwd" not in config:
        config["pwd"] = "."

    if "meta" in config:
        group = config["meta"]["group"]
        name = config["meta"]["name"]
        args.dest_arg = (config["meta"]["dest-arg"] == "yes")
        print(config)
    else:
        group = args.group
        name = args.name
    ckpt_root = args.ckpt_root
    group_root = join(ckpt_root, group)

    if not IS_NEW_CLUSTER:
        if "cpus-per-task" in config["machine"]:
            sbatch += "--cpu %s " % config["machine"]["cpus-per-task"]
        if "mem" in config["machine"]:
            sbatch += "--mem %s " % config["machine"]["mem"]
        elif "mem-per-cpu" in config["machine"]:
            assert "cpus-per-task" in config["machine"]
            mem_cpu = int(config["machine"]["mem-per-cpu"].replace("G", ""))
            cpus = int(config["machine"]["cpus-per-task"])
            sbatch += "--mem %d " % (mem_cpu * cpus)
        if "gres" in config["machine"] and config["machine"]["gres"].startswith("gpu:"):
            ngpu = int(config["machine"]["gres"].split(":")[1])
            if ngpu >= 1:
                sbatch += "--gpu %d " % ngpu
                sbatch = sbatch.replace("fblearner_ash_cpuram_default", "fblearner_ash_bigbasin_fair")

    expdir = join(group_root, name)
    log_dir = join(expdir, "logs")
    continuing = False
    if os.path.exists(log_dir):
        print(f"Experiment {group_root}/{name} already exists. You can: ")
        print(f"- Run it anyway. Any output of a previous experiment with the same hyperparameters will be overwritten. The code used will be the existing code and not a fresh clone.")
        print(f"- Trash existing experiment folder and create a fresh one. The (old) experiment folder will be put in the trash ({args.trash_dir})")
        print(f"- (default) Abort. Experiment folder will be kpet untouched and new experiments will not be run.")
        print(f"What do you choose ? [run|trash|abort]")
        answer = input().lower()
        if answer == "run":
            continuing = True
        elif answer == "trash":
            trash_expe(expdir, args.trash_dir)
            os.makedirs(log_dir)
        else:
            import sys;sys.exit(0)
    else:
        os.makedirs(log_dir)

    expdir = join(group_root, name)
    if not continuing:
        if args.grid != expdir + ".json":
            shutil.copyfile(args.grid, join(expdir, "sweep.json"))

        if IS_NEW_CLUSTER:
            if "git" in config:
                if "commit" in config and len(config["commit"]) > 0:
                    run_command("cd %s && git clone --depth 1 --single-branch --branch %s %s code" % (expdir, config["commit"], config["git"]))
                else:
                    run_command("cd %s && git clone %s code" % (expdir, config["git"]))
                path_repo = join(expdir, "code")
                if os.path.exists(join(path_repo, "init.sh")):
                    run_command("cd %s && ./init.sh" % path_repo)
                config["pwd"] = join(expdir, "code", config["pwd"])

        else:
            assert "git" in config
            # Make a fresh download of the code and send it to gfsai
            tmpdir = tempfile.NamedTemporaryFile().name
            assert tmpdir.startswith("/tmp")
            run_command("cd /tmp && git clone %s %s" % (config["git"], tmpdir[5:]))
            if os.path.exists(join(tmpdir, "init.sh")):
                run_command("cd %s && ./init.sh" % tmpdir)
            run_command("cd %s && tar cf code.tar . && cp code.tar %s" % (tmpdir, expdir))
            run_command("rm -rf %s" % tmpdir)
    else:
        config["pwd"] = join(expdir, "code", config["pwd"])


    # List config of parameters
    paramset = enumerateParams(config["params"])
    param_names = list(set([k for d in paramset for k in d.keys()]))
    param_values = {k: list(dict.fromkeys([(tuple(d[k]) if isinstance(d[k], list) else d[k]) for d in paramset if k in d])) for k in param_names}
    # param_values = {k: sorted(list(dict.fromkeys([(tuple(d[k]) if isinstance(d[k], list) else d[k]) for d in paramset if k in d]))) for k in param_names}
    # param_values = {k: sorted(list(set([d[k] for d in paramset if k in d]))) for k in param_names}
    if args.sample != -1:
        paramset = [paramset[i] for i in np.random.choice(len(paramset), args.sample, replace=False)]

    if os.path.exists(args.jobnames):
        jobnames = loadlist(args.jobnames)
        if group != '':
            jobnames.append("%s/%s" % (group, name))
        else:
            jobnames.append("%s" % (name))
        savelist(args.jobnames, jobnames)

    if args.launch_parallel:
        launchfilename = tempfile.NamedTemporaryFile().name
        launchfile = open(launchfilename, "w")

    params_to_index = [k for k, values in param_values.items() if len(values) >= 2 and (any(["/" in v for v in values if type(v) is str]) or any([len(str(v)) >= 20 for v in values]))]
    if args.array:
        filename = join(expdir, "run.sh")
        log_stdout = join(log_dir, "common.stdout")
        log_stderr = join(log_dir, "common.stderr")
        with open(filename, "w") as f:
            f.write(cmdPre(config, None, name, log_stdout, log_stderr, filename, num_expes=len(paramset), pooling=args.pooling))

        with open(join(expdir, "params.txt"), "w") as f, open(join(expdir, "commands.txt"), "w") as f_cmd, open(join(expdir, "status.txt"), "w") as f_status:
            for params in paramset:
                ext = generateExt(params, param_values, to_index=params_to_index)
                params = replaceMacros(params)

                log_stdout = join(log_dir, ext + ".stdout")
                log_stderr = join(log_dir, ext + ".stderr")

                if args.dest_arg:
                    if os.path.exists(join(expdir, ext)):
                        print('WARNING: %s already exists (whatever is in there will probably be overwritten by experiment)' % join(expdir, ext))
                    else:
                        os.makedirs(join(expdir, ext))
                    dest_name = config["meta"]["dest-name"] if "dest-name" in config["meta"] else "dest"
                    dest_name = [dest_name] if type(dest_name) is str else dest_name
                    for dname in dest_name:
                        params[dname] = join(expdir, ext)

                linear_params = linearize_params(params)

                f.write(f"{ext}\t{log_stdout}\t{log_stderr}\t{linear_params}\n")
                f_cmd.write(f"{config['cmd']} {linear_params}\n")
                f_status.write(f"{ext}\tPENDING\n")

        # print("To launch, execute: ")
        n_chunks = int(ceil(len(paramset) / args.pooling))
        cmd = sbatch + f"--array=1-{n_chunks} {filename}"
        print(cmd)

        if args.launch:
            r = subprocess.check_output([cmd], shell=True)
            print(r)
        else:
            print("Chmoding %s" % group_root)
            subprocess.check_output(["chmod -R a+w %s" % expdir], shell=True)
    else:
        for i_param, params in enumerate(paramset):
            ext = generateExt(params, param_values, to_index=params_to_index)
            if args.numeric:
                new_ext = "%d" % i_param
                with open(join(log_dir, new_ext + "_params.txt"), "w") as f:
                    f.write(ext)
                ext = new_ext

            params = replaceMacros(params)

            log_stdout = join(log_dir, ext + ".stdout")
            log_stderr = join(log_dir, ext + ".stderr")

            if args.dest_arg:
                if os.path.exists(join(expdir, ext)):
                    print('WARNING: %s already exists (whatever is in there will probably be overwritten by experiment)' % join(expdir, ext))
                else:
                    os.makedirs(join(expdir, ext))
                dest_name = config["meta"]["dest-name"] if "dest-name" in config["meta"] else "dest"
                dest_name = [dest_name] if type(dest_name) is str else dest_name
                for dname in dest_name:
                    params[dname] = join(expdir, ext)

            filename = join(expdir, 'run%s.sh' % ext)
            with open(filename, 'w') as f:
                f.write(cmdPre(config, params, name+ext, log_stdout, log_stderr, filename))

            if args.launch_parallel:
                launchfile.write(sbatch + filename + "&\n")
            else:
                print([sbatch + filename])
                start = time.time()
                if args.launch:
                    r = subprocess.check_output([sbatch + filename], shell=True)
                    # jobid = int(r.rstrip().split(" ")[3])
                    print(r)
                print("Took %.2f" % (time.time() - start))
                print(log_stdout)

        if not args.launch:
            print("Chmoding %s" % group_root)
            subprocess.check_output("chmod -R a+w %s" % group_root)

        if args.launch_parallel:
            launchfile.write("wait")
            launchfile.close()
            if args.launch:
                r = subprocess.check_output("bash %s" % launchfilename, shell=True)

elif args.command == 'count':
    config = json.load(open(args.grid), object_pairs_hook=OrderedDict)
    paramset = enumerateParams(config["params"])

    print("There are %d jobs in this sweep" % len(paramset))
elif args.command == 'remove':
    trash_expe(args.expe, args.trash_dir)
elif args.command == "list":
    print(run_command("ls " + join(args.ckpt_root, args.group)))
# Shamelessly copy-pasted from stool
elif args.command == 'stress':
    if args.partition != "":
        args.partition = " --partition=%s " % args.partition
    stool_stress(args.partition)
elif args.command == "status":
    users = [os.environ.get('USER')]
    users = ','.join(users)
    print(run_command('squeue -u ' + users + ' -o "%.18i %.50j %.9P %.2t %.10M %.6D %R" --sort=j'))
elif args.command == "notebook":
    assert os.path.exists(args.json_path)
    assert not os.path.exists(args.nb_path)
elif args.command == "check":
    assert os.path.exists(args.jobnames), "jobnames file not found at {}".format(args.jobnames)
    jobnames = loadlist(args.jobnames)
    remaining_jobnames = []
    jobnames = list(set(jobnames)) # deduplicating
    update = ""
    for jobname in jobnames:
        expdir = join(ckpt_default(), jobname)
        df = pd.read_csv(join(expdir, "status.txt"), delimiter='\t', names=['snippet', 'status'])
        df['new_status'] = df.apply(partial(update_status, expdir), axis=1)
        if step_number(df['new_status']) > step_number(df['status']):
            update += jobname + "\n"
            update += str(df)
        df['status'] = df['new_status']
        df = df.drop(columns=['new_status'])
        df.to_csv(join(ckpt_default(), jobname, "status.txt"), sep='\t', header=False, index=False)
        if np.any(df['status'] == 'PENDING') or np.any(df['status'] == 'RUNNING'):
            remaining_jobnames.append(jobname)
        
    if update != "":
        with open("/tmp/clutil_update.txt", "w") as f_update:
            f_update.write(update)
    savelist(args.jobnames, remaining_jobnames)
elif args.command == "relaunch":
    jobname = args.jobname
    assert os.path.exists(join(ckpt_default(), jobname))
    df = pd.read_csv(join(ckpt_default(), jobname, "status.txt"), delimiter='\t', names=['snippet', 'status'])
    job_ids = df.index[df["status"] == "FAILED"]
    for snippet in df["snippet"][job_ids].tolist():
        cmd = "rm " + join(ckpt_default(), jobname, "logs", snippet + '.std*')
        print(cmd)
        run_command(cmd)

    if len(job_ids) == 0:
        print("No jobs to relaunch")
        exit(0)
    cmd = f"sbatch --array={','.join([str(i+1) for i in job_ids])} {join(ckpt_default(), jobname, 'run.sh')}"
    print("Running the following command:")
    print(cmd)
    run_command(cmd)
    print(f"{len(job_ids)} jobs relaunched")
    df["status"] = df["status"].replace(['FAILED'], 'PENDING')
    df.to_csv(join(ckpt_default(), jobname, "status.txt"), sep='\t', header=False, index=False)


