# An Extensible Benchmark Suite for Learning to Simulate Physical Systems

This repository stores the code associated with our paper, to appear
at NeurIPS 2021 in the datasets and benchmarks track. A copy of our
paper is available from [OpenReview][openreview].

The code in this repository can be used to recreate our experiments
from our archived run description files, or modified to run additional
experiments either on the same data used in our work, or on modified
datasets with different compositions.

For a copy of the archived data as used for our tests, see our record
in the [NYU Faculty Digital Archive][nyuarchive]. For information on
using these records, see the supplementary material in our NeurIPS
paper which discusses the structure of the stored datasets. We have
also made available electronic descriptions of the experiments we ran
which can be used with the code here to rerun our experiments, or use
our stored network weights for further analysis without retraining.

## Contents

We include here the code we used in our experiments to:
1. Generate new data from our implemented simulations
2. Train neural networks for time derivative or step prediction
3. Test performance of the trained network on held-out data sets
4. Manage batches of jobs in each experiment

It can be used as-is to either recreate our experiments, or modified
to include new learning methods, new simulations, or adjustments to
problem settings.

See below for information on how to install the required dependencies
and run the software.

## Installation

After downloading this code, in order to run the software you will
need to install the necessary dependencies. We use
[Anaconda][anaconda] to manage most of the dependencies for this
project. The [environment.yml](environment.yml) lists these
dependencies. These include several proprietary dependencies such as
Nvidia's CUDA toolkit and Intel's MKL. Please review the licenses for
the external dependencies.

Additionally, generating data for the Navier-Stokes system requires a
built copy of [PolyFEM][polyfem].

There are two methods for configuring the required dependencies: as a
Singularity container, or manually as a Conda environment.

### Singularity Container

The simplest way to configure both of these components is to use a
[Singularity][singularity] container. We include a recipe for this in
[nn-benchmark.def](nn-benchmark.def). The command below should produce
an `nn-benchmark.sif` file containing the built container image:
```console
$ singularity build --fakeroot nn-benchmark.sif nn-benchmark.def
```
Documentation on [building containers][sbuild] is available from the
Singularity project.

### Manual Environment Setup

Alternatively, the environment and required dependencies can be
configured manually. First, create the Conda environment with the
required dependencies:
```console
$ conda env create -f environment.yml
```
For more information, see the documentation on [creating
environments][envcreate] with Conda.

The above step will create an environment `nn-benchmark` with the
required Python dependencies and interpreter. To activate it run:
```console
$ conda activate nn-benchmark
```

In order to generate new data for the Navier-Stokes system, you must
also build a copy of PolyFEM In a separate directory, clone the
[PolyFEM repository][polyfem]. To build it you will need a recent C++
compiler, CMake, a suitable build backend (such as Make), and a copy
of [Intel's MKL][mkl]. Once these are installed, in the copy of the
PolyFEM repository you cloned above run:
```console
$ mkdir build
$ cd build
$ MKLROOT=/path/to/mkl/root/ cmake .. -DPOLYSOLVE_WITH_PARDISO=ON -DPOLYFEM_NO_UI=ON
$ make
```
This will produce a binary `PolyFEM_bin` in the build directory. Once
you have this binary, either place it in the same directory as the
rest of the code, or ensure that it is available on the path.

Alternatively, you can set the environment variable `POLYFEM_BIN_DIR`
to the containing folder. The Navier-Stokes system will then use this
environment variable to locate the PolyFEM executable.

## Running

This software provides facilities for generating new datasets from the
implemented numerical simulators, and also for training and testing
neural networks on numerical simulation tasks.

The flow of experiment runs is divided into three phases: data
generation (`data_gen`), training (`train`), and evaluation (`eval`).
Descriptions of tasks in each of these phases are written to JSON
files, and run script runs the code with appropriate arguments, either
locally or by submitting jobs to a SLURM queue.

There are two entry point scripts which can be used to run these
tasks: `main.py` and `manage_runs.py`. The script `main.py` performs
the actual work of a single task: generating data, training a neural
network, or running an evaluation phase. An experiment run is composed
of many such tasks and these individual jobs are managed by
`manage_runs.py`. For single runs `main.py` can be used directly, but
for larger experiments `manage_runs.py` provides useful management
facilities.

### Run Descriptions

Each experiment has its own directory tree. Each task's parameters and
arguments are stored in a series of JSON files in a series of
directories: `descr/{data_gen,train,eval}/`. Each JSON file produces a
corresponding output directory under `run/{data_gen,train,eval}`.

Each JSON file contains a large number of arguments controlling the
execution of the job, describing the required resources (for SLURM
submission), and affecting the generation of data or training
configuration.

These files are created by "run generator" scripts which use utilities
objects in `run_generators/utils.py`. To illustrate this usage, the
run generation scripts used to produce the experiments run in our
paper are included in this repository. These can be run directly, and
take a single argument: a path to the folder where they will write
their output. Once the experiment directory has been populated by the
JSON task descriptions, the launcher script can be used to run each
phase. Be advised that running these scripts will resample random
parameters and so will produce data sets drawn from the same
distribution but with different contents.

### Main Script

The script `main.py` is responsible for performing the work of a
single job. It requires two arguments: the path to the JSON run
description file for it to follow and a path to the root of the
associated experiment directory. This second argument is necessary
because all loaded paths are relative to this root, which allows
relocating the experiments to different file systems.

For example, with an experiment generated under `experiment/` the
command below will run the job described in `description.json`.
```console
$ python main.py experiment/descr/{data_gen,train,eval}/description.json experiment/
```
This command needs to be run with the associated dependencies
available. Either run this command inside the Singularity container or
with the `nn-benchmark` environment loaded and environment variables
configured.

### Managing Runs

Running the individual jobs of an experiment individually with the
main script is possible, but unwieldy for large experiments. The
launcher script `manage_runs.py` provides useful facilities for
launching batches of jobs and managing their outputs.

#### Scanning

The first function of `manage_runs.py` is to check the status of a
directory of experiments and to report on the status of jobs found
there. The state of each job is reported as one of four categories:
1. Outstanding - The job has yet to be run (may be queued)
2. Mismatched - The input run description for this job was modified after it was run
3. Incomplete - The job is either still running or crashed
4. Finished - The job finished successfully

To check the status of the runs:
```console
$ python manage_runs.py scan experiment/
```

The scan utility produces a report on the status of the jobs, and
lists other issues that it detects with the experiment runs. In
particular, it detects issues with two jobs sharing the same output
directory.

The scan utility can also delete runs which exhibit these issues. Add
either `--delete=mismatch` or `--delete=incomplete` to the scan
command to delete the outputs of runs in these two states. This will
allow them to be relaunched.

*Warning:* There is **no** confirmation for the delete operation. Wait
until all running jobs have finished before using the delete
functionality and confirm that you want to delete the outputs before
adding the `--delete` option.

#### Submitting Jobs

The job management script submits batches of jobs. It is capable of
running them either locally (one after another on the local machine)
or by submitting them to a SLURM queue. When you are ready to launch
all pending jobs from one phase of the experiment:
```console
$ python manage_runs.py launch experiment/ data_gen
$ python manage_runs.py launch experiment/ train
$ python manage_runs.py launch experiment/ eval
```
After launching one phase, wait for all of its jobs to complete before
launching the next.

The script automatically detects whether a SLURM queue is available by
looking for the `sbatch` executable on the path. You can override this
auto-detection using the `--launch-type` argument with options:
`local`, `slurm`, or `auto` (the default).

The job management script itself requires only modules from the Python
standard library. However, running the jobs requires the rest of the
project dependencies to be available.

If you are using the Singularity container the management script will
look for the file `nn-benchmark.sif` in the current directory, and
next in a directory set in a `SCRATCH` environment variable. If the
container is found, and jobs are being submitted to a SLURM queue, the
container will be used automatically. If the container is being used,
the job launching script may warn that the conda environment is not
loaded. This can be ignored as the container will provide the
environment for each running job.

In other cases, you must load the Conda environment before running the
job launching script. Ensure that the `nn-benchmark` Conda environment
is loaded and available.

Consult `manage_runs.py --help` more information on available options.

## Citing

If you make use of this software, please cite our associated paper:
```bibtex
@article{nnbenchmark21,
  title={An Extensible Benchmark Suite for Learning to Simulate Physical Systems},
  author={Karl Otness and Arvi Gjoka and Joan Bruna and Daniele Panozzo and Benjamin Peherstorfer and Teseo Schneider and Denis Zorin},
  year={2021},
  url={https://openreview.net/forum?id=pY9MHwmrymR}
}
```

## License

This software is made available under the terms of the MIT license.
See [LICENSE.txt](LICENSE.txt) for details.

The external dependencies used by this software are available under a
variety of different licenses. Please review these external licenses.

[envcreate]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
[openreview]: https://openreview.net/forum?id=pY9MHwmrymR
[nyuarchive]: https://archive.nyu.edu/handle/2451/63285
[polyfem]: https://github.com/polyfem/polyfem/
[anaconda]: https://www.anaconda.com/
[singularity]: https://singularity.hpcng.org/
[sbuild]: https://singularity.hpcng.org/user-docs/3.8/build_a_container.html
[mkl]: https://software.intel.com/oneapi/onemkl
