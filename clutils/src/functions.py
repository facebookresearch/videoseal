import os
import subprocess
import textwrap
import random
import argparse
from math import ceil
dirname = os.path.dirname

IS_FAIR_CLUSTER = os.path.exists("/checkpoint")
IS_AWS_CLUSTER = os.path.exists("/fsx-checkpoints")
IS_NEW_CLUSTER = IS_FAIR_CLUSTER or IS_AWS_CLUSTER

FALSY_STRINGS = ['off', 'false', '0']
TRUTHY_STRINGS = ['on', 'true', '1']


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def replaceMacros(params):
    assert type(params) is dict
    new_params = {}
    for k, v in params.items():
        if type(v) is str and v.startswith('__SAMPLE__('):
            pos_begin = len('__SAMPLE__(')
            pos_end = v.find(')')

            a, b = [int(x) for x in v[pos_begin:pos_end].split(",")]
            new_params[k] = random.randint(a, b)
        elif type(v) is str and v == '__SEED__()':
            new_params[k] = int.from_bytes(
                os.urandom(8), byteorder="big", signed=True)
        else:
            new_params[k] = v

    return new_params


def run_command(command):
    output = subprocess.check_output(command, shell=True)
    return output.decode('ascii')


def getuser():
    return run_command("whoami").rstrip()


def getName(jobid):
    r = subprocess.check_output(
        ["sacct --format=jobname%%200 -j %d" % jobid], shell=True)

    return r.split("\n")[2].rstrip().lstrip()


def getStatus(jobid):
    r = subprocess.check_output(["sacct -j %d" % jobid], shell=True)
    for s in r.split("\n"):
        if s.startswith("%d " % jobid):
            return s.split()[-2]


def savelist(fname, x):
    with open(fname, "w") as f:
        for val in x:
            f.write("%s\n" % val)


def loadlist(fname):
    l = []
    with open(fname) as f:
        for line in f:
            l.append(line.rstrip())

    return l


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def cmdPre(config, params, name, log_stdout, log_stderr, run_filename, pooling=1, num_expes=-1):
    if IS_NEW_CLUSTER:
        return cmdPreNew(config, params, name, log_stdout, log_stderr, run_filename, pooling=pooling, num_expes=num_expes)
    else:
        return cmdPreOld(config, params, name, log_stdout, log_stderr, run_filename)


def cmdPreOld(config, params, name, log_stdout, log_stderr, run_filename):
    s = ""
    s += 'LOG_STDOUT="%s"\n' % log_stdout
    s += 'LOG_STDERR="%s"\n' % log_stderr
    s += textwrap.dedent("""\
    code_dir=$(mktemp -d)
    cd $code_dir

    cp /mnt/vol/gfsai-flash-east/ai-group/users/asablayrolles/binary_embeddings/201807.tar .
    tar xf 201807.tar

    mkdir code
    cd code
    cp {expdir}/code.tar .
    tar xf code.tar

    """).format(expdir=dirname(run_filename))

    if "pwd" in config:
        s += "cd %s\n" % config["pwd"]

    if "cat" in config:
        for catfile in config["cat"]:
            s += "cat %s >> $LOG_STDOUT\n" % catfile
    if "precmd" in config:
        for cmd in config["precmd"]:
            s += "%s\n" % cmd
    s += "which python >> $LOG_STDOUT\n"
    s += 'echo "---Beginning program ---" >> $LOG_STDOUT\n'
    s += "PYTHONUNBUFFERED=yes "
    if "preload" in config:
        s += config["preload"] + " "

    s += "%s \\\n" % config["cmd"]

    for k, v in params.items():
        if len(k) == 1:
            s += "-%s %s " % (k, v)
        elif k.isupper():
            s += "%s %s " % (k, v)
        else:
            s += "--%s %s " % (k, v)
    s += ">> $LOG_STDOUT 2>> $LOG_STDERR\n"

    return s


def linearize_params(params):
    s = ""
    for k, v in params.items():
        if isinstance(v, list):
            v = ' '.join([str(x) for x in v])
        if len(k) == 1:
            s += f"-{k} {v} "
        elif k.isupper():
            s += f"{k} {v} "
        else:
            if v == "__true__":
                s += f"--{k} "
            elif v == "__false__":
                s += f"--no-{k} "
            else:
                s += f"--{k} {v} "

    return s


def cmdPreNew(config, params, name, log_stdout, log_stderr, run_filename, pooling=1, num_expes=-1):
    pwd = config["pwd"]
    params_filename = run_filename.replace("run.sh", "params.txt")
    s = "#!/bin/bash\n\n"
    for k, v in config["machine"].items():
        if v == "":
            s += f"#SBATCH --{k}\n"
        else:
            s += f"#SBATCH --{k}={v}\n"

    s += textwrap.dedent(f"""\
    #SBATCH --output={log_stdout}
    #SBATCH --error={log_stderr}
    #SBATCH --job-name={name}
    #SBATCH --open-mode=append
    #SBATCH --signal=B:USR1@120

    cd {pwd}
    """)

    if pooling > 1:
        assert num_expes != -1
        last_expe = num_expes + 1
        step = int(ceil(num_expes / pooling))
        s += f"for((EXP_NUMBER=$SLURM_ARRAY_TASK_ID; EXP_NUMBER<{last_expe}; EXP_NUMBER+={step})); do\n"
        s += textwrap.dedent(f"""\
        if [ "$EXP_NUMBER" -gt "$(wc -l {params_filename} | cut -f1 -d' ')" ]; then
            break
        fi
        """)
    else:
        s += "EXP_NUMBER=$SLURM_ARRAY_TASK_ID\n"

    # For a job array
    if params is None:
        s += textwrap.dedent(f"""\
        export JOBNAME=$(sed "${{EXP_NUMBER}}q;d" {params_filename} | cut -f1)
        LOG_STDOUT=$(sed "${{EXP_NUMBER}}q;d" {params_filename} | cut -f2)
        LOG_STDERR=$(sed "${{EXP_NUMBER}}q;d" {params_filename} | cut -f3)
        PARAMS=$(sed "${{EXP_NUMBER}}q;d" {params_filename} | cut -f4)

        trap_handler () {{
           echo "Caught signal" >> $LOG_STDOUT
           sbatch --begin=now+120 {run_filename}
           exit 0
        }}
        """)
    else:
        s += textwrap.dedent(f"""\
        export JOBNAME="{name}"
        LOG_STDOUT="{log_stdout}"
        LOG_STDERR="{log_stderr}"

        trap_handler () {{
           echo "Caught signal" >> $LOG_STDOUT
           sbatch --begin=now+120 {run_filename}
           exit 0
        }}
        """)

    s += textwrap.dedent(f"""\
    function ignore {{
       echo "Ignored SIGTERM" >> $LOG_STDOUT
    }}

    trap ignore TERM
    trap trap_handler USR1
    echo "Git hash:" >> $LOG_STDOUT
    echo $(git rev-parse HEAD 2> /dev/null) >> $LOG_STDOUT

    """)  # .format(run_filename=run_filename)

    if "cat" in config:
        for catfile in config["cat"]:
            s += f"cat {catfile} >> $LOG_STDOUT\n"

    if "precmd" in config:
        for cmd in config["precmd"]:
            s += f"{cmd}\n"

    s += "which python >> $LOG_STDOUT\n"
    s += 'echo "---Beginning program ---" >> $LOG_STDOUT\n'
    s += 'start_time=$SECONDS'
    s += "PYTHONUNBUFFERED=yes "
    if "preload" in config:
        s += config["preload"] + " "

    s += "srun %s \\\n" % config["cmd"]

    if params is None:
        s += "$PARAMS"
    else:
        s += linearize_params(params)

    # s += ">> $LOG_STDOUT 2>> $LOG_STDERR && echo 'CLUTIL_JOB_FINISHED' >> $LOG_STDOUT \n"
    s += ">> $LOG_STDOUT 2>> $LOG_STDERR && echo 'CLUTIL_JOB_FINISHED' >> $LOG_STDOUT && echo 'CLUTIL_TIME' $(( SECONDS - start_time )) >> $LOG_STDOUT &\n"
    s += "wait $!\n"
    if pooling > 1:
        s += "done"

    return s
