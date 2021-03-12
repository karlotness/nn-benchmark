# An Extensible Benchmark Suite for Learning to Simulate Physical Systems

This is the code implementing the systems considered in the paper as
well as to run the experiments whose results are reported.

**Note:** The code we include here is preliminary. We plan to do
additional cleanup, and to produce a more polished code repository and
additional information for users, as well as maintaining this as we
extend the benchmark.

## Dependencies

We use Anaconda to manage the dependencies for this project. The
`environment.yml` file lists what is required. Use `conda` to
[create the environment][envcreate]

This will create an `nn-benchmark` environment which can be activated.

Alternatively, the `nn-benchmark.def` file can be used to create a
Singularity container containing the Anaconda environment.

## Running

The flow of experiment runs is divided into three phases: data
generation (`data_gen`), training (`train`), and evaluation (`eval`).
Descriptions of tasks in each of these phases are written to JSON
files, and run script runs the code with appropriate arguments, either
locally or by submitting jobs to a SLURM queue.

### Run Descriptions

Each experiment is stored in its own directory. Creating the JSON
files is handled by separate "run generator" scripts. Scripts which
produce the experiments reported in the paper are in the
`run_generators/` folder. These can be run directly, and take a single
argument: a path to the folder where they will write their output.
Once the experiment directory has been populated by the JSON task
descriptions, the launcher script can be used to run each phase.

### Launcher

Running the tasks themselves requires the Anaconda environment to be
available. Either, it should be already activated before running
`manage_runs.py`, *or* the Singularity container should be available.
The launcher searches for the `nn-benchmark.sif` file in a path
specified by the `SCRATCH` environment variable. If found, the
Singularity container will be used.

To check the status of the runs:
```
python manage_runs.py scan <path/to/experiment_directory>
```
The scan utility can also delete failed runs. Check the `--help`
option for more information.

When you are ready to launch a phase of the experiment
```
python manage_runs.py launch <path/to/experiment_directory> data_gen
python manage_runs.py launch <path/to/experiment_directory> train
python manage_runs.py launch <path/to/experiment_directory> eval
```
after launching one phase, wait for all its jobs to complete before
launching the next.

Consult `manage_runs.py --help` more information on available options.

[envcreate]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
