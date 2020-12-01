import copy
import numpy as np
import pathlib
import json


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


class SpringInitialConditionSource(InitialConditionSource):
    def __init__(self, radius_range=(0.2, 1)):
        super().__init__()
        self.radius_range = radius_range

    def _sample_ring_uniform(self, inner_r, outer_r, num_pts=1):
        theta = np.random.uniform(0, 2*np.pi, num_pts)
        unifs = np.random.uniform(size=num_pts)
        r = np.sqrt(unifs * (outer_r**2 - inner_r**2) + inner_r**2)
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
                 noise_sigma=0.0):
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
                "time": "10:00:00",
                "cpus": 8 if self.gpu else 32,
                "mem": 64 if self.training_set.system == "wave" else 32,
            },
        }
        if self.validation_set is not None:
            template["phase_args"]["train_data"]["val_data_dir"] = self.validation_set.path
        if self.gpu:
            template["phase_args"]["train_data"]["loader"].update({
                "pin_memory": True,
                "num_workers": 8,
            })
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
                "time": "10:00:00",
                "cpus": 8 if self.gpu else 32,
                "mem": 64 if self.training_set.system == "wave" else 32,
            },
        }
        if self.validation_set is not None:
            template["phase_args"]["train_data"]["val_data_dir"] = self.validation_set.path
        if self.gpu:
            template["phase_args"]["train_data"]["loader"].update({
                "pin_memory": True,
                "num_workers": 8,
            })
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
                "cpus": 8 if self.gpu else 32,
                "mem": 64 if self.training_set.system == "wave" else 32,
            },
        }
        if self.validation_set is not None:
            template["phase_args"]["train_data"]["val_data_dir"] = self.validation_set.path
        if self.gpu:
            template["phase_args"]["train_data"]["loader"].update({
                "pin_memory": True,
                "num_workers": 8,
            })
        return template


class MLP(TrainedNetwork):
    def __init__(self, experiment, training_set, gpu=True, learning_rate=1e-3,
                 hidden_dim=2048, depth=2, train_dtype="float",
                 batch_size=750, epochs=1000, validation_set=None):
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
        self._check_val_set(train_set=self.training_set, val_set=self.validation_set)

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
                "time": "10:00:00",
                "cpus": 8 if self.gpu else 32,
                "mem": 64 if self.training_set.system == "wave" else 32,
            },
        }
        if self.validation_set is not None:
            template["phase_args"]["train_data"]["val_data_dir"] = self.validation_set.path
        if self.gpu:
            template["phase_args"]["train_data"]["loader"].update({
                "pin_memory": True,
                "num_workers": 8,
            })
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
                "mem": 64 if self.training_set.system == "wave" else 32,
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
                "mem": 64 if self.training_set.system == "wave" else 32,
            },
        }
        return template


class Evaluation(WritableDescription):
    def __init__(self, experiment, name_tail):
        super().__init__(experiment=experiment,
                         phase="eval",
                         name=f"eval-{name_tail}")


class NetworkEvaluation(Evaluation):
    def __init__(self, experiment, network, eval_set, gpu=False, integrator="leapfrog",
                 eval_dtype=None):
        super().__init__(experiment=experiment,
                         name_tail=f"net-{network.name}-set-{eval_set.name}-{integrator}")
        self.network = network
        self.eval_set = eval_set
        self.gpu = gpu
        if eval_dtype is None:
            self.eval_dtype = self.network.train_dtype
        else:
            self.eval_dtype = eval_dtype
        self.integrator = integrator
        if self.network.method == "knn-predictor":
            self.integrator = "null"
        # Validate inputs
        assert isinstance(self.network, TrainedNetwork)
        if self.eval_set.system != self.network.training_set.system:
            raise ValueError(f"Inconsistent systems {self.eval_set.system} and {self.network.training_set.system}")

    def description(self):
        eval_type = self.network.method
        gpu = self.gpu and (eval_type not in {"knn-predictor", "knn-regressor"})
        template = {
            "phase_args": {
                "eval_net": self.network.path,
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
                "time": "04:00:00",
                "cpus": 16,
                "mem": 64 if self.eval_set.system == "wave" else 32,
            },
        }
        return template


class BaselineIntegrator(Evaluation):
    def __init__(self, experiment, eval_set, integrator="leapfrog",
                 eval_dtype="double"):
        super().__init__(experiment=experiment,
                         name_tail=f"integrator-baseline-{eval_set.name}-{integrator}")
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
                "mem": 64 if self.eval_set.system == "wave" else 32,
            },
        }
        return template
