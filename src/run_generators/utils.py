import copy
import numpy as np
import pathlib
import json
import math
import re
import dataclasses
import itertools


def generate_packing_args(instance, system, dataset):
    if system == "spring":
        dim = dataset.input_size() // 2
        assert dim == 1
        instance.particle_process_type = "one-dim"
        instance.adjacency_args = {
            "type": "regular-grid",
            "boundary_conditions": "fixed",
            "boundary_vertices": [[-1., 0.], [1., 0.]],
            "dimension": dim,
            }
        instance.v_features = 4
        instance.e_features = 6
        instance.mesh_coords = [[-1., 0.], [0., 0.], [1., 0.]]
        instance.static_nodes = [1, 0, 1]
    elif system == "wave":
        dim = dataset.input_size() // 2
        instance.particle_process_type = "one-dim"
        instance.adjacency_args = {
            "type": "regular-grid",
            "boundary_conditions": "periodic",
            "boundary_vertices": None,
            "dimension": dim,
            }
        instance.v_features = 4
        instance.e_features = 6
        instance.mesh_coords = [[x, 0] for x in np.linspace(0, 1, dim)]
        instance.static_nodes = [0 for i in np.arange(dim)]
    elif system == "spring-mesh":
        dim = dataset.input_size() // 2
        instance.particle_process_type = "identity"
        instance.adjacency_args = {
            "type": "native",
            "boundary_conditions": None,
            "boundary_vertices": None,
            "dimension": dim,
            }
        instance.v_features = 4
        instance.e_features = 6
        instance.mesh_coords = [list(map(float, p["position"])) for p in dataset.initial_cond_source.particle_properties()]
        instance.static_nodes = [1 if p["is_fixed"] else 0 for p in dataset.initial_cond_source.particle_properties()]
    else:
        raise ValueError(f"Invalid system {system}")


def generate_scheduler_args(instance, end_lr):
    def gamma_factor(initial_lr, final_lr, epochs):
        return np.power(final_lr / initial_lr, 1. / epochs)

    if end_lr is not None and instance.scheduler == "exponential":
        instance.scheduler_args = {
            "gamma": gamma_factor(
                instance.learning_rate, end_lr, instance.epochs),
        }
    else:
        instance.scheduler_args = None


class Experiment:
    def __init__(self, name):
        self.name = name
        self._name_counters = {}

    def _get_run_suffix(self, name):
        if name not in self._name_counters:
            self._name_counters[name] = 0
        self._name_counters[name] += 1
        return self._name_counters[name]

    def get_run_name(self, name_core):
        suffix = self._get_run_suffix(name=name_core)
        name = f"{self.name}_{name_core}_{suffix:05}"
        return name

    @staticmethod
    def get_name_core(name):
        match = re.match(r"^[^_]+_(?P<name>.+?)_\d{5}$", name)
        if match is None:
            raise ValueError(f"Invalid name format: {name}")
        else:
            return match.group("name")


class InitialConditionSource:
    def __init__(self):
        self._initial_conditions = []

    def _generate_initial_condition(self):
        raise NotImplementedError("Subclass this")

    def sample_initial_conditions(self, num):
        if num > len(self._initial_conditions):
            remaining = num - len(self._initial_conditions)
            for _i in range(remaining):
                new_cond = self._generate_initial_condition()
                self._initial_conditions.append(new_cond)
        return [copy.deepcopy(d) for d in self._initial_conditions[:num]]


class WaveInitialConditionSource(InitialConditionSource):
    def __init__(self,
                 height_range=(0.75, 1.25),
                 width_range=(0.75, 1.25),
                 position_range=(0.5, 0.5)):
        super().__init__()
        self.height_range = height_range
        self.width_range = width_range
        self.position_range = position_range

    def _generate_initial_condition(self):
        width = np.random.uniform(*self.width_range)
        height = np.random.uniform(*self.height_range)
        position = np.random.uniform(*self.position_range)
        state = {
            "start_type": "cubic_splines",
            "start_type_args": {
                "width": width,
                "height": height,
                "position": position,
            }
        }
        return state


class WaveDisjointInitialConditionSource(WaveInitialConditionSource):
    def __init__(self,
                 height_range=((0.75, 1.25),),
                 width_range=((0.75, 1.25),),
                 position_range=((0.5, 0.5),)):
        super().__init__(height_range=height_range,
                         width_range=width_range,
                         position_range=position_range)
        for ranges in [self.height_range, self.width_range,
                       self.position_range]:
            for a, b in ranges:
                assert a <= b

    def __in_range(self, value, ranges):
        for minv, maxv in ranges:
            if minv <= value <= maxv:
                return True
        return False

    def __rejection_sample_value(self, ranges):
        minv = min(a for a, _b in ranges)
        maxv = max(b for _a, b in ranges)
        while True:
            value = np.random.uniform(minv, maxv)
            if self.__in_range(value, ranges):
                return value

    def _generate_initial_condition(self):
        width = self.__rejection_sample_value(self.width_range)
        height = self.__rejection_sample_value(self.height_range)
        position = self.__rejection_sample_value(self.position_range)
        state = {
            "start_type": "cubic_splines",
            "start_type_args": {
                "width": width,
                "height": height,
                "position": position,
            }
        }
        return state


class SpringInitialConditionSource(InitialConditionSource):
    def __init__(self, radius_range=(0.2, 1)):
        super().__init__()
        self.radius_range = radius_range

    def _sample_ring_uniform(self, inner_r, outer_r, num_pts=1):
        theta = np.random.uniform(0, 2*np.pi, num_pts)
        r = np.random.uniform(low=inner_r, high=outer_r, size=num_pts)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack((x, y), axis=-1)

    def _generate_initial_condition(self):
        pt = self._sample_ring_uniform(*self.radius_range)[0]
        p = pt[0].item()
        q = pt[1].item()
        state = {
            "initial_condition": {
                "q": q,
                "p": p,
            },
        }
        return state


class ParticleInitialConditionSource(InitialConditionSource):
    def __init__(self, n_particles, n_dim, scale=1.0, masses=None):
        super().__init__()
        self.n_particles = n_particles
        self.n_dim = n_dim
        self.scale = scale
        if masses is None:
            self.masses = np.ones(self.n_particles)
        else:
            self.masses = masses
        assert len(self.masses.shape) == 1
        assert self.masses.shape[0] == self.n_particles

    def _generate_initial_condition(self):
        p0 = np.random.normal(scale=self.scale, size=(self.n_particles, self.n_dim))
        q0 = np.random.normal(scale=self.scale, size=(self.n_particles, self.n_dim))
        state = {
            "p0": p0.tolist(),
            "q0": q0.tolist(),
            "masses": self.masses.tolist(),
        }
        return state


class SpringMeshGridGenerator:
    def __init__(self, grid_shape):
        self.grid_shape = grid_shape
        self.n_dims = len(grid_shape)
        self.n_dim = self.n_dims
        self.n_particles = 1
        for s in grid_shape:
            self.n_particles *= s
        self._particles = None
        self._springs = None

    def generate_mesh(self):
        if self._particles is None:
            particles = []
            springs = []
            ranges = [range(s) for s in self.grid_shape]
            # Generate particle descriptions
            for coords in itertools.product(*ranges):
                fixed = all(i == 0 or i == m - 1 for i, m in zip(coords, self.grid_shape))
                particle_def = {
                    "mass": 1.0,
                    "is_fixed": fixed,
                    "position": list(coords),
                }
                particles.append(particle_def)
            # Add edges
            for (a, part_a), (b, part_b) in itertools.combinations(enumerate(particles), 2):
                # Determine if we want an edge
                dist = max(abs(pa - pb) for pa, pb in zip(part_a["position"], part_b["position"]))
                if dist != 1:
                    continue
                # Add the edge
                length = math.sqrt(sum((pa - pb) ** 2 for pa, pb in zip(part_a["position"], part_b["position"])))
                spring_def = {
                    "a": a,
                    "b": b,
                    "spring_const": 1.0,
                    "rest_length": length,
                }
                springs.append(spring_def)
            self._particles = particles
            self._springs = springs
        return copy.deepcopy(self._particles), copy.deepcopy(self._springs)


class SpringMeshManualPerturb(InitialConditionSource):
    def __init__(self, mesh_generator, perturbations):
        super().__init__()
        self.mesh_generator = mesh_generator
        self.perturbations = perturbations
        self.n_dim = mesh_generator.n_dim
        self.n_particles = mesh_generator.n_particles

    def particle_properties(self):
        particles, _springs = self.mesh_generator.generate_mesh()
        return particles

    def _generate_initial_condition(self):
        particles, springs = self.mesh_generator.generate_mesh()
        # Apply perturbation
        for particle in particles:
            for target_coord, perturb in self.perturbations:
                if all(p == tc for p, tc in zip(particle["position"], target_coord)):
                    # Apply perturbation
                    particle["position"] = [pos + diff for pos, diff in zip(particle["position"], perturb)]
        return {
            "particles": particles,
            "springs": springs,
        }


class WritableDescription:
    def __init__(self, experiment, phase, name):
        self.experiment = experiment
        self.name = name
        self.full_name = self.experiment.get_run_name(name)
        self.path = f"run/{phase}/{self.full_name}"
        self._descr_path = f"descr/{phase}/{self.full_name}.json"
        self.phase = phase

    def description(self):
        # Subclasses provide top-level dictionary including slurm_args
        # So dictionary with "phase_args" and "slurm_args" keys
        # Rest is filled in here
        raise NotImplementedError("Subclass this")

    def write_description(self, base_dir):
        base_dir = pathlib.Path(base_dir)
        out_path = base_dir / pathlib.Path(self._descr_path)
        # Construct core values
        descr = {
            "out_dir": self.path,
            "phase": self.phase,
            "exp_name": self.experiment.name,
            "run_name": self.full_name,
        }
        # Update values for experiment
        descr.update(self.description())
        # Write description to file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf8") as out_file:
            json.dump(descr, out_file)


class Dataset(WritableDescription):
    def __init__(self, experiment, name_tail, system, set_type="train"):
        super().__init__(experiment=experiment,
                         phase="data_gen",
                         name=f"{set_type}-{system}-{name_tail}")
        self.set_type = set_type
        self.system = system

    def data_dir(self):
        return self.path

    def input_size(self):
        raise NotImplementedError("Subclass this")


class SpringDataset(Dataset):
    def __init__(self, experiment, initial_cond_source, num_traj,
                 set_type="train",
                 num_time_steps=30, time_step_size=0.3, rtol=1e-10,
                 noise_sigma=0.0,
                 mesh_based=False):
        super().__init__(experiment=experiment,
                         name_tail=f"n{num_traj}-t{num_time_steps}-n{noise_sigma}",
                         system="spring",
                         set_type=set_type)
        self.num_traj = num_traj
        self.initial_cond_source = initial_cond_source
        self.num_time_steps = num_time_steps
        self.time_step_size = time_step_size
        self.rtol = rtol
        self.noise_sigma = noise_sigma
        self.initial_conditions = self.initial_cond_source.sample_initial_conditions(self.num_traj)
        self.mesh_based = mesh_based
        assert isinstance(self.initial_cond_source, SpringInitialConditionSource)

    def description(self):
        # Build trajectories
        trajectories = []
        for icond in self.initial_conditions:
            traj = {
                "num_time_steps": self.num_time_steps,
                "time_step_size": self.time_step_size,
                "rtol": self.rtol,
                "noise_sigma": self.noise_sigma
            }
            traj.update(icond)
            trajectories.append(traj)
        template = {
            "phase_args": {
                "system": "spring",
                "system_args": {
                    "trajectory_defs": trajectories
                }
            },
            "slurm_args": {
                "gpu": False,
                "time": "00:30:00",
                "cpus": 8,
                "mem": 6,
            },
        }
        return template

    def input_size(self):
        return 2


class WaveDataset(Dataset):
    def __init__(self, experiment, initial_cond_source, num_traj,
                 set_type="train", n_grid=250,
                 num_time_steps=200, time_step_size=0.1, wave_speed=0.1,
                 subsampling=1000, noise_sigma=0.0):
        super().__init__(experiment=experiment,
                         name_tail=f"n{num_traj}-t{num_time_steps}-n{noise_sigma}",
                         system="wave",
                         set_type=set_type)
        self.space_max = 1
        self.n_grid = n_grid
        self.wave_speed = wave_speed
        self.subsampling = subsampling
        self.num_traj = num_traj
        self.initial_cond_source = initial_cond_source
        self.num_time_steps = num_time_steps
        self.time_step_size = time_step_size
        self.noise_sigma = noise_sigma
        self.initial_conditions = self.initial_cond_source.sample_initial_conditions(self.num_traj)
        assert isinstance(self.initial_cond_source, WaveInitialConditionSource)

    def description(self):
        trajectories = []
        for icond in self.initial_conditions:
            traj = {
                "wave_speed": self.wave_speed,
                "num_time_steps": self.num_time_steps,
                "time_step_size": self.time_step_size,
                "subsample": self.subsampling,
                "noise_sigma": self.noise_sigma,
            }
            traj.update(icond)
            trajectories.append(traj)
        # Generate template
        template = {
            "phase_args": {
                "system": "wave",
                "system_args": {
                    "space_max": self.space_max,
                    "n_grid": self.n_grid,
                    "trajectory_defs": trajectories,
                }
            },
            "slurm_args": {
                "gpu": False,
                "time": "05:00:00",
                "cpus": 16,
                "mem": 64,
            },
        }
        return template

    def input_size(self):
        return 2 * self.n_grid


class ParticleDataset(Dataset):
    def __init__(self, experiment, initial_cond_source, num_traj,
                 set_type="train", n_dim=2, n_particles=2,
                 num_time_steps=200, time_step_size=0.1, noise_sigma=0.0,
                 g=1.0, rtol=1e-6):
        super().__init__(experiment=experiment,
                         name_tail=f"n{num_traj}-t{num_time_steps}-n{noise_sigma}",
                         system="particle",
                         set_type=set_type)
        self.num_traj = num_traj
        self.num_time_steps = num_time_steps
        self.time_step_size = time_step_size
        self.rtol = rtol
        self.noise_sigma = noise_sigma
        self.n_particles = n_particles
        self.n_dim = n_dim
        self.g = g
        self.initial_cond_source = initial_cond_source
        self.initial_conditions = self.initial_cond_source.sample_initial_conditions(self.num_traj)
        assert isinstance(self.initial_cond_source, ParticleInitialConditionSource)
        assert self.initial_cond_source.n_particles == self.n_particles
        assert self.initial_cond_source.n_dim == self.n_dim

    def description(self):
        trajectories = []
        for icond in self.initial_conditions:
            traj = {
                "num_time_steps": self.num_time_steps,
                "time_step_size": self.time_step_size,
                "rtol": self.rtol,
                "noise_sigma": self.noise_sigma,
            }
            traj.update(icond)
            trajectories.append(traj)
        template = {
            "phase_args": {
                "system": "particle",
                "system_args": {
                    "n_particles": self.n_particles,
                    "n_dim": self.n_dim,
                    "g": self.g,
                    "trajectory_defs": trajectories,
                }
            },
            "slurm_args": {
                "gpu": False,
                "time": "00:30:00",
                "cpus": 8,
                "mem": 6,
            },
        }
        return template

    def input_size(self):
        return 2 * self.n_dim * self.n_particles


class SpringMeshDataset(Dataset):
    def __init__(self, experiment, initial_cond_source, num_traj,
                 set_type="train",
                 num_time_steps=500, time_step_size=0.1,
                 subsampling=10, noise_sigma=0.0, vel_decay=1.0):
        super().__init__(experiment=experiment,
                         name_tail=f"n{num_traj}-t{num_time_steps}-n{noise_sigma}",
                         system="spring-mesh",
                         set_type=set_type)
        self.initial_cond_source = initial_cond_source
        self.num_traj = num_traj
        self.num_time_steps = num_time_steps
        self.time_step_size = time_step_size
        self.subsampling = subsampling
        self.noise_sigma = noise_sigma
        self.vel_decay = vel_decay
        self.initial_conditions = self.initial_cond_source.sample_initial_conditions(self.num_traj)
        self.n_dim = initial_cond_source.n_dim
        self.n_particles = initial_cond_source.n_particles

    def description(self):
        trajectories = []
        for icond in self.initial_conditions:
            traj = {
                "num_time_steps": self.num_time_steps,
                "time_step_size": self.time_step_size,
                "noise_sigma": self.noise_sigma,
                "subsample": self.subsampling,
            }
            traj.update(icond)
            trajectories.append(traj)
        template = {
            "phase_args": {
                "system": "spring-mesh",
                "system_args": {
                    "vel_decay": self.vel_decay,
                    "trajectory_defs": trajectories,
                }
            },
            "slurm_args": {
                "gpu": False,
                "time": "03:00:00",
                "cpus": 8,
                "mem": 32,
            },
        }
        return template

    def input_size(self):
        return 2 * self.n_dim * self.n_particles


class ExistingDataset:
    def __init__(self, descr_path):
        self._descr_path = pathlib.Path(descr_path)
        # Load the data
        with open(self._descr_path, 'r', encoding='utf8') as in_file:
            self._descr = json.load(in_file)
        # Extract relevant measures
        self.system = self._descr["phase_args"]["system"]
        self.path = self._descr["out_dir"]
        self.name = Experiment.get_name_core(self._descr["run_name"])
        # Handle system-specific values
        if self.system == "spring":
            self._input_size = 2
        elif self.system == "wave":
            self.n_grid = self._descr["phase_args"]["system_args"]["n_grid"]
            self._input_size = 2 * self.n_grid
        elif self.system == "particle":
            # Handle particle-specific values
            self.n_particles = self._descr["phase_args"]["system_args"]["n_particles"]
            self.n_dim = self._descr["phase_args"]["system_args"]["n_dim"]
            self._input_size = 2 * self.n_dim * self.n_particles
        else:
            raise ValueError(f"Unknown system {self.system}")
        # Check some basic values
        assert self._descr["phase"] == "data_gen"

    def input_size(self):
        return self._input_size

    def data_dir(self):
        return self.path


class TrainedNetwork(WritableDescription):
    def __init__(self, experiment, method, name_tail):
        super().__init__(experiment=experiment,
                         phase="train",
                         name=f"{method}-{name_tail}")
        self.method = method

    def _check_val_set(self, train_set, val_set):
        if val_set is None:
            return
        assert val_set.system == train_set.system
        assert val_set.input_size() == train_set.input_size()

    def _get_mem_requirement(self, train_set):
        system = train_set.system
        if system == "wave":
            if train_set.num_traj > 100 and train_set.num_time_steps > 8500:
                return 73
            else:
                return 37
        return 16


class HNN(TrainedNetwork):
    def __init__(self, experiment, training_set, gpu=True, learning_rate=1e-3,
                 output_dim=2, hidden_dim=200, depth=3, train_dtype="float",
                 field_type="solenoidal", batch_size=750,
                 epochs=1000, validation_set=None):
        super().__init__(experiment=experiment,
                         method="hnn",
                         name_tail=f"{training_set.name}-d{depth}-h{hidden_dim}")
        self.training_set = training_set
        self.gpu = gpu
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.train_dtype = train_dtype
        self.field_type = field_type
        self.batch_size = batch_size
        self.validation_set = validation_set
        self._check_val_set(train_set=self.training_set, val_set=self.validation_set)


    def description(self):
        template = {
            "phase_args": {
                "network": {
                    "arch": "hnn",
                    "arch_args": {
                        "base_model": "mlp",
                        "input_dim": self.training_set.input_size(),
                        "base_model_args": {
                            "hidden_dim": self.hidden_dim,
                            "output_dim": self.output_dim,
                            "depth": self.depth,
                            "nonlinearity": "tanh",
                        },
                        "hnn_args": {
                            "field_type": self.field_type,
                        },
                    },
                },
                "training": {
                    "optimizer": "adam",
                    "optimizer_args": {
                        "learning_rate": self.learning_rate,
                    },
                    "max_epochs": self.epochs,
                    "try_gpu": self.gpu,
                    "train_dtype": self.train_dtype,
                    "train_type": "hnn",
                    "train_type_args": {},
                },
                "train_data": {
                    "data_dir": self.training_set.path,
                    "dataset": "snapshot",
                    "linearize": True,
                    "dataset_args": {},
                    "loader": {
                        "batch_size": self.batch_size,
                        "shuffle": True,
                    },
                },
            },
            "slurm_args": {
                "gpu": self.gpu,
                "time": "15:00:00",
                "cpus": 8 if self.gpu else 20,
                "mem": self._get_mem_requirement(train_set=self.training_set),
            },
        }
        if self.validation_set is not None:
            template["phase_args"]["train_data"]["val_data_dir"] = self.validation_set.path
        return template


class SRNN(TrainedNetwork):
    def __init__(self, experiment, training_set, gpu=True, learning_rate=1e-3,
                 output_dim=2, hidden_dim=2048, depth=2, train_dtype="float",
                 batch_size=750, epochs=1000, rollout_length=10,
                 integrator="leapfrog", validation_set=None):
        super().__init__(experiment=experiment,
                         method="srnn",
                         name_tail=f"{training_set.name}-d{depth}-h{hidden_dim}-i{integrator}")
        self.training_set = training_set
        self.gpu = gpu
        self.learning_rate = learning_rate
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.train_dtype = train_dtype
        self.batch_size = batch_size
        self.epochs = epochs
        self.rollout_length = rollout_length
        self.integrator = integrator
        self.validation_set = validation_set
        self._check_val_set(train_set=self.training_set, val_set=self.validation_set)

    def description(self):
        template = {
            "phase_args": {
                "network": {
                    "arch": "srnn",
                    "arch_args": {
                        "base_model": "mlp",
                        "input_dim": self.training_set.input_size() // 2,
                        "hidden_dim": self.hidden_dim,
                        "output_dim": self.output_dim,
                        "depth": self.depth,
                        "nonlinearity": "tanh",
                    },
                },
                "training": {
                    "optimizer": "adam",
                    "optimizer_args": {
                        "learning_rate": self.learning_rate,
                    },
                    "max_epochs": self.epochs,
                    "try_gpu": self.gpu,
                    "train_dtype": self.train_dtype,
                    "train_type": "srnn",
                    "train_type_args": {
                        "rollout_length": self.rollout_length,
                        "integrator": self.integrator,
                    },
                },
                "train_data": {
                    "data_dir": self.training_set.path,
                    "dataset": "rollout-chunk",
                    "linearize": True,
                    "dataset_args": {
                        "rollout_length": self.rollout_length,
                    },
                    "loader": {
                        "batch_size": self.batch_size,
                        "shuffle": True,
                    },
                },
            },
            "slurm_args": {
                "gpu": self.gpu,
                "time": "15:00:00",
                "cpus": 8 if self.gpu else 20,
                "mem": self._get_mem_requirement(train_set=self.training_set),
            },
        }
        if self.validation_set is not None:
            template["phase_args"]["train_data"]["val_data_dir"] = self.validation_set.path
        return template


class HOGN(TrainedNetwork):
    def __init__(self, experiment, training_set, gpu=True, hidden_dim=64,
                 connection_radius=5, learning_rate=1e-3, epochs=300,
                 train_dtype="float", batch_size=100, validation_set=None):
        super().__init__(experiment=experiment,
                         method="hogn",
                         name_tail=f"{training_set.name}-h{hidden_dim}")
        self.training_set = training_set
        self.hidden_dim = hidden_dim
        self.gpu = gpu
        self.connection_radius = connection_radius
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.train_dtype = train_dtype
        self.batch_size = batch_size
        self.validation_set = validation_set
        self._check_val_set(train_set=self.training_set, val_set=self.validation_set)
        # Infer values from training set
        if self.training_set.system == "wave":
            self.particle_process_type = "one-dim"
            self.adjacency_args = {
                "type": "circular-local",
                "dimension": self.training_set.input_size() // 2,
                "degree": connection_radius,
            }
            self.input_dim = 3
            self.ndim = 1
        elif self.training_set.system == "spring":
            self.particle_process_type = "one-dim"
            self.adjacency_args = {
                "type": "fully-connected",
                "dimension": self.training_set.input_size() // 2,
            }
            self.input_dim = 3
            self.ndim = 1
        elif self.training_set.system == "particle":
            self.particle_process_type = "identity"
            self.adjacency_args = self.adjacency_args = {
                "type": "fully-connected",
                "dimension": self.training_set.n_particles
            }
            self.ndim = self.training_set.n_dim
            self.input_dim = 2 * self.ndim + 1
        else:
            raise ValueError(f"Invalid system {self.training_set.system}")

    def description(self):
        template = {
            "phase_args": {
                "network": {
                    "arch": "hogn",
                    "arch_args": {
                        "input_dim": self.input_dim,
                        "ndim": self.ndim,
                        "hidden_dim": self.hidden_dim,
                    },
                },
                "training": {
                    "optimizer": "adam",
                    "optimizer_args": {
                        "learning_rate": self.learning_rate,
                    },
                    "max_epochs": self.epochs,
                    "try_gpu": self.gpu,
                    "train_dtype": self.train_dtype,
                    "train_type": "hogn",
                    "train_type_args": {},
                },
                "train_data": {
                    "data_dir": self.training_set.path,
                    "dataset": "snapshot",
                    "dataset_args": {},
                    "loader": {
                        "type": "pytorch-geometric",
                        "batch_size": self.batch_size,
                        "shuffle": True,
                        "package_args": {
                            "particle_processing": self.particle_process_type,
                            "package_type": "hogn",
                            "adjacency_args": self.adjacency_args,
                        },
                    },
                },
            },
            "slurm_args": {
                "gpu": self.gpu,
                "time": "14:00:00",
                "cpus": 8 if self.gpu else 20,
                "mem": self._get_mem_requirement(train_set=self.training_set),
            },
        }
        if self.validation_set is not None:
            template["phase_args"]["train_data"]["val_data_dir"] = self.validation_set.path
        return template


class GN(TrainedNetwork):
    def __init__(self, experiment, training_set, gpu=True, hidden_dim=128,
                 learning_rate=1e-4, end_lr=1e-6, epochs=300, layer_norm=False,
                 scheduler="exponential", scheduler_step="epoch",
                 noise_type="none", noise_variance=0,
                 train_dtype="float", batch_size=100, validation_set=None):
        super().__init__(experiment=experiment,
                         method="gn",
                         name_tail=f"{training_set.name}-h{hidden_dim}")
        self.training_set = training_set
        self.hidden_dim = hidden_dim
        self.gpu = gpu
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layer_norm = layer_norm
        self.train_dtype = train_dtype
        self.batch_size = batch_size
        self.validation_set = validation_set
        self.scheduler = scheduler
        self.scheduler_step = scheduler_step
        self.noise_type = noise_type
        self.noise_variance = noise_variance
        self._check_val_set(
            train_set=self.training_set, val_set=self.validation_set)
        generate_packing_args(
            self, self.training_set.system, self.training_set)
        generate_scheduler_args(self, end_lr)

    def description(self):
        template = {
            "phase_args": {
                "network": {
                    "arch": "gn",
                    "arch_args": {
                        "v_features": self.v_features,
                        "e_features": self.e_features,
                        "hidden_dim": self.hidden_dim,
                        "mesh_coords": self.mesh_coords,
                        "static_nodes": self.static_nodes,
                        "layer_norm": self.layer_norm,
                    },
                },
                "training": {
                    "optimizer": "adam",
                    "optimizer_args": {
                        "learning_rate": self.learning_rate,
                    },
                    "max_epochs": self.epochs,
                    "try_gpu": self.gpu,
                    "train_dtype": self.train_dtype,
                    "train_type": "gn",
                    "train_type_args": {
                    },
                    "scheduler": self.scheduler,
                    "scheduler_step": self.scheduler_step,
                    "scheduler_args": self.scheduler_args,
                },
                "train_data": {
                    "data_dir": self.training_set.path,
                    "dataset": "snapshot",
                    "dataset_args": {},
                    "loader": {
                        "type": "pytorch-geometric",
                        "batch_size": self.batch_size,
                        "shuffle": True,
                        "package_args": {
                            "particle_processing": self.particle_process_type,
                            "package_type": "gn",
                            "adjacency_args": self.adjacency_args,
                        },
                    },
                },
            },
            "slurm_args": {
                "gpu": self.gpu,
                "time": "14:00:00",
                "cpus": 8 if self.gpu else 20,
                "mem": self._get_mem_requirement(train_set=self.training_set),
            },
        }
        if self.validation_set is not None:
            template["phase_args"]["train_data"]["val_data_dir"] = self.validation_set.path
        if self.noise_type != "none":
            template["phase_args"]["training"]["noise"] = {
                "type": self.noise_type,
                "variance": self.noise_variance
            }
        return template




class MLP(TrainedNetwork):
    def __init__(self, experiment, training_set, gpu=True, learning_rate=1e-3,
                 hidden_dim=2048, depth=2, train_dtype="float",
                 scheduler="none", scheduler_step="epoch", end_lr=None,
                 batch_size=750, epochs=1000, validation_set=None,
                 noise_type="none", noise_variance=0):
        super().__init__(experiment=experiment,
                         method="mlp",
                         name_tail=f"{training_set.name}-d{depth}-h{hidden_dim}")
        self.training_set = training_set
        self.gpu = gpu
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.train_dtype = train_dtype
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_set = validation_set
        self.scheduler = scheduler
        self.scheduler_step = scheduler_step
        self._check_val_set(train_set=self.training_set, val_set=self.validation_set)
        self.noise_type = noise_type
        self.noise_variance = noise_variance
        generate_scheduler_args(self, end_lr)

    def description(self):
        template = {
            "phase_args": {
                "network": {
                    "arch": "mlp",
                    "arch_args": {
                        "input_dim": self.training_set.input_size(),
                        "hidden_dim": self.hidden_dim,
                        "output_dim": self.training_set.input_size(),
                        "depth": self.depth,
                        "nonlinearity": "tanh",
                    },
                },
                "training": {
                    "optimizer": "adam",
                    "optimizer_args": {
                        "learning_rate": self.learning_rate,
                    },
                    "max_epochs": self.epochs,
                    "try_gpu": self.gpu,
                    "train_dtype": self.train_dtype,
                    "train_type": "mlp",
                    "train_type_args": {},
                    "scheduler": self.scheduler,
                    "scheduler_step": self.scheduler_step,
                    "scheduler_args": self.scheduler_args,
                },
                "train_data": {
                    "data_dir": self.training_set.path,
                    "dataset": "snapshot",
                    "linearize": True,
                    "dataset_args": {},
                    "loader": {
                        "batch_size": self.batch_size,
                        "shuffle": True,
                    },
                },
            },
            "slurm_args": {
                "gpu": self.gpu,
                "time": "15:00:00",
                "cpus": 8 if self.gpu else 20,
                "mem": self._get_mem_requirement(train_set=self.training_set),
            },
        }
        if self.validation_set is not None:
            template["phase_args"]["train_data"]["val_data_dir"] = self.validation_set.path
        if self.noise_type != "none":
            template["phase_args"]["training"]["noise"] = {
                "type": self.noise_type,
                "variance": self.noise_variance
            }
        return template


class NNKernel(TrainedNetwork):
    def __init__(self, experiment, training_set, gpu=True, learning_rate=1e-3,
                 hidden_dim=2048, train_dtype="float",
                 batch_size=750, epochs=1000, validation_set=None,
                 nonlinearity="relu", optimizer="sgd", weight_decay=0):
        super().__init__(experiment=experiment,
                         method="nn-kernel",
                         name_tail=f"{training_set.name}-h{hidden_dim}-lr{learning_rate}-wd{weight_decay}")
        self.training_set = training_set
        self.gpu = gpu
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.train_dtype = train_dtype
        self.batch_size = batch_size
        self.epochs = epochs
        self.nonlinearity = nonlinearity
        self.validation_set = validation_set
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self._check_val_set(train_set=self.training_set, val_set=self.validation_set)

    def description(self):
        template = {
            "phase_args": {
                "network": {
                    "arch": "nn-kernel",
                    "arch_args": {
                        "input_dim": self.training_set.input_size(),
                        "hidden_dim": self.hidden_dim,
                        "output_dim": self.training_set.input_size(),
                        "nonlinearity": self.nonlinearity,
                    },
                },
                "training": {
                    "optimizer": self.optimizer,
                    "optimizer_args": {
                        "learning_rate": self.learning_rate,
                        "weight_decay": self.weight_decay,
                    },
                    "max_epochs": self.epochs,
                    "try_gpu": self.gpu,
                    "train_dtype": self.train_dtype,
                    "train_type": "nn-kernel",
                    "train_type_args": {},
                },
                "train_data": {
                    "data_dir": self.training_set.path,
                    "dataset": "snapshot",
                    "linearize": True,
                    "dataset_args": {},
                    "loader": {
                        "batch_size": self.batch_size,
                        "shuffle": True,
                    },
                },
            },
            "slurm_args": {
                "gpu": self.gpu,
                "time": "15:00:00",
                "cpus": 8 if self.gpu else 20,
                "mem": self._get_mem_requirement(train_set=self.training_set),
            },
        }
        if self.validation_set is not None:
            template["phase_args"]["train_data"]["val_data_dir"] = self.validation_set.path
        return template


class KNNRegressor(TrainedNetwork):
    def __init__(self, experiment, training_set):
        super().__init__(experiment=experiment,
                         method="knn-regressor",
                         name_tail=f"{training_set.name}")
        self.training_set = training_set
        self.train_dtype = "double"

    def description(self):
        template = {
            "phase_args": {
                "network": {
                    "arch": "knn-regressor",
                    "arch_args": {},
                },
                "training": {
                    "max_epochs": 0,
                    "try_gpu": False,
                    "train_dtype": self.train_dtype,
                    "train_type": "knn-regressor",
                    "train_type_args": {},
                },
                "train_data": {
                    "data_dir": self.training_set.path,
                    "dataset": "snapshot",
                    "linearize": True,
                    "dataset_args": {},
                    "loader": {
                        "batch_size": 750,
                        "shuffle": False,
                    },
                },
            },
            "slurm_args": {
                "gpu": False,
                "time": "01:30:00",
                "cpus": 8,
                "mem": self._get_mem_requirement(train_set=self.training_set),
            },
        }
        return template


class KNNPredictor(TrainedNetwork):
    def __init__(self, experiment, training_set):
        super().__init__(experiment=experiment,
                         method="knn-predictor",
                         name_tail=f"{training_set.name}")
        self.training_set = training_set
        self.train_dtype = "double"

    def description(self):
        template = {
            "phase_args": {
                "network": {
                    "arch": "knn-predictor",
                    "arch_args": {},
                },
                "training": {
                    "max_epochs": 0,
                    "try_gpu": False,
                    "train_dtype": self.train_dtype,
                    "train_type": "knn-predictor",
                    "train_type_args": {},
                },
                "train_data": {
                    "data_dir": self.training_set.path,
                    "dataset": "snapshot",
                    "linearize": True,
                    "dataset_args": {},
                    "loader": {
                        "batch_size": 750,
                        "shuffle": False,
                    },
                },
            },
            "slurm_args": {
                "gpu": False,
                "time": "01:30:00",
                "cpus": 8,
                "mem": self._get_mem_requirement(train_set=self.training_set),
            },
        }
        return template


class ExistingNetwork:
    def __init__(self, descr_path, root_dir=None):
        self._descr_path = pathlib.Path(descr_path)
        if root_dir is None:
            root_dir = self._descr_path.parent.parent.parent
        # Load the data
        with open(self._descr_path, 'r', encoding='utf8') as in_file:
            self._descr = json.load(in_file)
        # Extract relevant measures
        self.path = self._descr["out_dir"]
        self.name = Experiment.get_name_core(self._descr["run_name"])
        # Create data set
        # Hack: get name of the description from output dir
        train_set_path = list(pathlib.Path(self._descr["phase_args"]["train_data"]["data_dir"]).parts)
        train_set_path[0] = "descr"
        train_set_path[-1] = train_set_path[-1] + ".json"
        train_set_path = pathlib.Path(*train_set_path)
        self.training_set = ExistingDataset(root_dir / train_set_path)
        # Handle system-specific details
        self.method = self._descr["phase_args"]["network"]["arch"]
        self.train_dtype = self._descr["phase_args"]["training"]["train_dtype"]
        # Check some basic values
        assert self._descr["phase"] == "train"


class Evaluation(WritableDescription):
    def __init__(self, experiment, name_tail):
        super().__init__(experiment=experiment,
                         phase="eval",
                         name=f"eval-{name_tail}")

    def _get_mem_requirement(self, eval_set):
        system = eval_set.system
        if system == "wave":
            return 32
        return 16


class NetworkEvaluation(Evaluation):
    def __init__(self, experiment, network, eval_set, gpu=False, integrator=None,
                 eval_dtype=None, network_file="model.pt"):
        if network.method in {"knn-predictor", "knn-predictor-oneshot", "gn"}:
            integrator = "null"
        if integrator is None:
            raise ValueError("Must manually specify integrator")
        super().__init__(experiment=experiment,
                         name_tail=f"net-{network.name}-set-{eval_set.name}-{integrator}")
        self.network = network
        self.network_file = network_file
        self.eval_set = eval_set
        self.gpu = gpu
        if eval_dtype is None:
            self.eval_dtype = self.network.train_dtype
        else:
            self.eval_dtype = eval_dtype
        self.integrator = integrator
        # Validate inputs
        if self.eval_set.system != self.network.training_set.system:
            raise ValueError(f"Inconsistent systems {self.eval_set.system} and {self.network.training_set.system}")

        if self.network.method == "gn":
            system = eval_set.system
            generate_packing_args(self, system, self.eval_set)


    def description(self):
        eval_type = self.network.method
        gpu = self.gpu and (eval_type not in {"knn-predictor", "knn-regressor", "knn-predictor-oneshot", "knn-regressor-oneshot"})
        template = {
            "phase_args": {
                "eval_net": self.network.path,
                "eval_net_file": self.network_file,
                "eval_data": {
                    "data_dir": self.eval_set.path,
                    "linearize": (self.network.method != "hogn"),
                },
                "eval": {
                    "eval_type": eval_type,
                    "integrator": self.integrator,
                    "eval_dtype": self.eval_dtype,
                    "try_gpu": gpu,
                }
            },
            "slurm_args": {
                "gpu": gpu,
                "time": "16:00:00",
                "cpus": 8 if gpu else 20,
                "mem": self._get_mem_requirement(eval_set=self.eval_set),
            },
        }
        if self.network.method == "gn":
            template["phase_args"]["eval"]["package_args"] = {
                "particle_processing": self.particle_process_type,
                "package_type": "gn",
                "adjacency_args": self.adjacency_args,
            }
        return template


class KNNOneshotEvaluation(NetworkEvaluation):
    MockNetwork = dataclasses.make_dataclass("MockNetwork", ["name", "train_dtype", "training_set", "method", "path"])

    def __init__(self, experiment, training_set, eval_set, knn_type,
                 eval_dtype="double", integrator="leapfrog"):
        method = f"knn-{knn_type}-oneshot"
        self._mock_network = self.MockNetwork(name=f"{method}-{training_set.name}",
                                              train_dtype=eval_dtype,
                                              training_set=training_set,
                                              method=method,
                                              path=None)
        super().__init__(experiment=experiment, network=self._mock_network,
                         eval_set=eval_set, gpu=False, integrator=integrator,
                         eval_dtype=eval_dtype)
        self.training_set = training_set

    def description(self):
        template = super().description()
        # Add arguments to construct the training set
        template["phase_args"]["eval"]["train_data"] = {
            "data_dir": self.training_set.path,
            "dataset": "snapshot",
            "linearize": True,
            "dataset_args": {},
            "loader": {
                "batch_size": 750,
                "shuffle": False,
            },
        }
        return template


class KNNPredictorOneshot(KNNOneshotEvaluation):
    def __init__(self, experiment, training_set, eval_set,
                 eval_dtype="double"):
        super().__init__(experiment=experiment,
                         training_set=training_set,
                         eval_set=eval_set,
                         eval_dtype=eval_dtype,
                         integrator="null",
                         knn_type="predictor")


class KNNRegressorOneshot(KNNOneshotEvaluation):
    def __init__(self, experiment, training_set, eval_set,
                 eval_dtype="double", integrator="leapfrog"):
        super().__init__(experiment=experiment,
                         training_set=training_set,
                         eval_set=eval_set,
                         eval_dtype=eval_dtype,
                         integrator=integrator,
                         knn_type="regressor")


class BaselineIntegrator(Evaluation):
    def __init__(self, experiment, eval_set, integrator="leapfrog",
                 eval_dtype="double"):
        super().__init__(experiment=experiment,
                         name_tail=f"integrator-baseline-{eval_set.name}-{integrator}-{eval_dtype}")
        self.eval_set = eval_set
        self.eval_dtype = eval_dtype
        self.integrator = integrator

    def description(self):
        template = {
            "phase_args": {
                "eval_net": None,
                "eval_data": {
                    "data_dir": self.eval_set.path,
                },
                "eval": {
                    "eval_type": "integrator-baseline",
                    "integrator": self.integrator,
                    "eval_dtype": self.eval_dtype,
                    "try_gpu": False,
                },
            },
            "slurm_args": {
                "gpu": False,
                "time": "04:00:00",
                "cpus": 16,
                "mem": self._get_mem_requirement(eval_set=self.eval_set),
            },
        }
        return template
