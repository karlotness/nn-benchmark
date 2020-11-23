import copy
import numpy as np
import pathlib
import json


class Experiment:
    def __init__(self, name):
        self.name = name
        self._counter = 0

    def _get_run_suffix(self):
        count = self._counter
        self._counter += 1
        return count

    def get_run_name(self, name_core):
        suffix = self._get_run_suffix()
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
            for _i in remaining:
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
        pt = self._sample_ring_uniform(*self.radius_range)
        p = pt[0].item()
        q = pt[1].item()
        state = {
            "initial_condition": {
                "q": q,
                "p": p,
            },
        }
        return state


class SavableDescription:
    def __init__(self, experiment, phase, name):
        self.experiment = experiment
        self.name = self.experiment.get_run_name(name)
        self.path = f"run/{phase}/{self.name}"
        self._descr_path = f"descr/{phase}/{self.name}.json"
        self.phase = phase

    def description(self):
        # Subclasses provide top-level dictionary including slurm_args
        # So dictionary with "phase_args" and "slurm_args" keys
        # Rest is filled in here
        raise NotImplementedError("Subclass this")

    def write_description(self, base_dir):
        base_dir = pathlib.Path(base_dir)
        out_path = base_dir / pathlib.Path(self._descr_path)
        descr = self.description()
        # Update core values
        descr.update({
            "out_dir": self.path,
            "phase": self.phase,
            "exp_name": self.experiment.name,
            "run_name": self.name,
        })
        # Write description to file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf8") as out_file:
            json.dump(descr, out_file)


class Dataset(SavableDescription):
    def __init__(self, experiment, name_tail, system, set_type="train"):
        super().__init__(experiment=experiment,
                         phase="data_gen",
                         name=f"{set_type}-{system}-{name_tail}")
        self.set_type = set_type
        self.system = system

    def data_dir(self):
        return self.path


class SpringDataset(Dataset):
    def __init__(self, experiment, initial_cond_source, num_traj,
                 set_type="train",
                 num_time_steps=30, time_step_size=0.3, rtol=1e-10,
                 noise_sigma=0.0):
        super().__init__(experiment=experiment,
                         name_tail=f"n{num_traj}-t{num_time_steps}",
                         system="spring",
                         set_type=set_type)
        self.num_traj = num_traj
        self.initial_cond_source = initial_cond_source
        self.num_time_steps = num_time_steps
        self.time_step_size = time_step_size
        self.rtol = rtol
        self.noise_sigma = noise_sigma
        self.initial_conditions = self.initial_cond_source.sample_initial_conditions(self.num_traj)
        assert isinstance(self.initial_conditions, SpringInitialConditionSource)

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


class WaveDataset(Dataset):
    def __init__(self, experiment, initial_cond_source, num_traj,
                 set_type="train", n_grid=250,
                 num_time_steps=200, time_step_size=0.1, wave_speed=0.1,
                 subsampling=1000, noise_sigma=0.0):
        super().__init__(experiment=experiment,
                         name_tail=f"n{num_traj}-t{num_time_steps}",
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
        assert isinstance(self.initial_conditions, WaveInitialConditionSource)

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
                "mem": 32,
            },
        }
        return template