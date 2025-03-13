## Overview

This Python script is designed to manage and monitor machine learning experiments on a cluster environment. It provides functionalities to launch, track, and manage experiments, including handling different statuses of jobs (e.g., pending, running, failed, finished), moving experiments to a trash directory, and checking the system's resource usage. The script supports both new and old cluster configurations and integrates with SLURM workload manager for job scheduling.

## Requirements

- Python 3.x
- NumPy
- Pandas
- SLURM (for job scheduling on clusters)
- Git (for version control and managing code repositories)
- Subprocess and OS modules (for interacting with the system)

## Usage

### Quickstart

**Sweep**

To launch a series of experiments based on a configuration grid, use the following command:
```bash
python main.py sweep <path_to_config_file>
```

This will clone the repo, go into the code directory and launch the experiments. 
The experiment directory will have with the following structure:
```
jobname/
├── code/
├── logs/
├───── jobname_0.stdout
├───── jobname_0.stderr
├───── ...
├── jobname_0/
├── jobname_1/
├── ...
├── run.sh
├── commands.txt
├── params.txt
```

**Check**

To check the status of jobs submitted by the user, use the following command (same as `squeue`):
```bash
python main.py status
```

### Alias

To avoid typing `python main.py` every time, you can create an alias in your `.bashrc` file:
```bash
echo "alias clutil='python ~/clutils/main.py'" >> ~/.bashrc && source ~/.bashrc
```

Then, you can use the following commands:
```bash
clutils sweep <path_to_config_file>
clutils status
```

### JSON file example

The configuration file should be in JSON format. Ex:
```json
{
   "args": {
      "arg1": ["val1", "val2"],
      "arg2": {
         "val1": {
            "arg3": ["val1"],
         },
         "val2": {
            "arg3": ["val2"],
         }
      }
      ...
   },
   "machine": {
      "partition": "partition_name",
      "time": "time_limit",
      ...
   },
   "meta": {
      "name": "jobname",
      ...
   }
}
```
This will generate the following grid:
```
arg1=val1, arg2=val1, arg3=val1
arg1=val1, arg2=val2, arg3=val2
arg1=val2, arg2=val1, arg3=val1
arg1=val2, arg2=val2, arg3=val2
```



### Commands and args

1. **sweep**: Launch a series of experiments based on a configuration grid. It handles job creation, parameter enumeration, and job submission.
   - `--grid`: Path to the configuration file.
   - `--no-launch`: Flag to prepare jobs without launching (default: launch).
   - `--sample`: Number of random configurations to sample from the grid (default: use all configurations).
   - `--numeric`: Use numeric identifiers for job names (default: use args as identifiers).
   - `--array`: Use job arrays for batch submission (default: array).
   - `--pooling`: Number of jobs to pool together.

2. **count**: Count the number of jobs in a sweep configuration.
   - `grid`: Path to the configuration file.

3. **remove**: Move an experiment directory to a specified trash directory.
   - `expe`: Path to the experiment directory.

4. **list**: List experiments in a group.
   - `group`: Name of the group.

5. **status**: Check the status of jobs submitted by the user.

6. **check**: Check and update the status of all jobs listed in the jobnames file.

7. **relaunch**: Relaunch failed jobs.
   - `jobname`: Name of the job to relaunch.

8. **cancel**: Cancel specified jobs.
   - `jobname`: Name of the job to cancel.

9. **stress**: Check the current resource usage on the cluster.
   - `--partition`: Specify the partition to check.


## Users and maintainers

- Pierre Fernandez (pfz)
- Tom Sander (tomsander)

Users:
- AV Seal team

Creator:
- Alex Sablayrolles