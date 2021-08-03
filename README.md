# An Extensible Benchmark Suite for Learning to Simulate Physical Systems

This repository stores the code associated with our paper, to appear
at NeurIPS 2021 in the datasets and benchmarks track. A copy of our
paper is available from [OpenReview][openreview].

The code in this repository can be used to recreate our experiments
from our archived run description files, or modified to run additional
experiments either on the same data used in our work, or on modified
datasets with different compositions.

For a copy of the archived data as used for our tests, see our record
in the [NYU Faculty Digital Archive][nyuarchive].

## Contents

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
you have this binary, set the environment variable `POLYFEM_BIN_DIR`
to the containing folder. The Navier-Stokes system will use this
environment variable to locate the PolyFEM executable.

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
[nyuarchive]: https://archive.nyu.edu/
[polyfem]: https://github.com/polyfem/polyfem/
[anaconda]: https://www.anaconda.com/
[singularity]: https://singularity.hpcng.org/
[sbuild]: https://singularity.hpcng.org/user-docs/3.8/build_a_container.html
[mkl]: https://software.intel.com/oneapi/onemkl
