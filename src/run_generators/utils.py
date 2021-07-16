import copy
import numpy as np
import pathlib
import json
import math
import re
import dataclasses
import itertools
from scipy import interpolate


class Experiment:
    def __init__(self, name):
        self.name = name
        self._name_counters = {}

    def _get_run_suffix(self, name, name_tag=None):
        name_key = (name, name_tag)
        if name_key not in self._name_counters:
            self._name_counters[name_key] = 0
        self._name_counters[name_key] += 1
        return self._name_counters[name_key]

    def get_run_name(self, name_core, name_tag=None):
        suffix = self._get_run_suffix(name=name_core, name_tag=name_tag)
        tag = ""
        if name_tag:
            tag = f"tag{name_tag}_"
        name = f"{self.name}_{name_core}_{tag}{suffix:05}"
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


class SpringMeshGridGenerator:
    def __init__(self, grid_shape, fix_particles="corners"):
        self.grid_shape = grid_shape
        self.n_dims = len(grid_shape)
        self.n_dim = self.n_dims
        self.n_particles = 1
        for s in grid_shape:
            self.n_particles *= s
        self._particles = None
        self._springs = None
        if fix_particles == "corners":
            self._fixed_pred = lambda self, coords: all(i == 0 or i == m - 1 for i, m in zip(coords, self.grid_shape))
        elif fix_particles == "top":
            self._fixed_pred = lambda self, coords: coords[1] == self.grid_shape[1] - 1

    def generate_mesh(self):
        if self._particles is None:
            particles = []
            springs = []
            ranges = [range(s) for s in self.grid_shape]
            # Generate particle descriptions
            for coords in itertools.product(*ranges):
                fixed = self._fixed_pred(self, coords)
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


class SpringMeshRowPerturb(InitialConditionSource):
    def __init__(self, mesh_generator, magnitude, row=0):
        super().__init__()
        self.mesh_generator = mesh_generator
        self.magnitude = magnitude
        self.row = row
        self.n_dim = mesh_generator.n_dim
        self.n_particles = mesh_generator.n_particles
        assert self.n_dim == 2

    def particle_properties(self):
        particles, _springs = self.mesh_generator.generate_mesh()
        return particles

    def _generate_initial_condition(self):
        angle = np.random.uniform(0, 2 * np.pi)
        perturb_x = np.cos(angle) * self.magnitude
        perturb_y = np.sin(angle) * self.magnitude
        particles, springs = self.mesh_generator.generate_mesh()
        # Apply perturbation
        for particle in particles:
            if particle["is_fixed"]:
                # Do not perturb fixed particles
                continue
            if particle["position"][1] == self.row:
                # Apply perturbation
                particle["position"][0] += perturb_x
                particle["position"][1] += perturb_y
        return {
            "particles": particles,
            "springs": springs,
        }


class SpringMeshInterpolatePerturb(InitialConditionSource):
    def __init__(self, mesh_generator, coords, magnitude_range=(0, 0.75)):
        super().__init__()
        self.mesh_generator = mesh_generator
        self.magnitude_range = magnitude_range
        self.coords = coords
        self.n_dim = mesh_generator.n_dim
        self.n_particles = mesh_generator.n_particles
        assert self.n_dim == 2

    def particle_properties(self):
        particles, _springs = self.mesh_generator.generate_mesh()
        return particles

    def _sample_ring_uniform(self, inner_r, outer_r, num_pts=1):
        theta = np.random.uniform(0, 2*np.pi, num_pts)
        unifs = np.random.uniform(size=num_pts)
        r = np.sqrt(unifs * (outer_r**2 - inner_r**2) + inner_r**2)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack((x, y), axis=-1)

    def _generate_initial_condition(self):
        perturbs = self._sample_ring_uniform(min(self.magnitude_range),
                                             max(self.magnitude_range),
                                             num_pts=len(self.coords))
        coord_arr = np.array(self.coords)
        pert_fx = interpolate.LinearNDInterpolator(coord_arr, perturbs[:, 0])
        pert_fy = interpolate.LinearNDInterpolator(coord_arr, perturbs[:, 1])

        particles, springs = self.mesh_generator.generate_mesh()
        # Apply perturbation
        for particle in particles:
            if particle["is_fixed"]:
                # Do not perturb fixed particles
                continue
            # Perturb this particle using interpolated values
            x = particle["position"][0]
            y = particle["position"][1]
            perturb_x = pert_fx(x, y)
            perturb_y = pert_fy(x, y)
            # Apply perturbation
            particle["position"][0] += perturb_x
            particle["position"][1] += perturb_y
        return {
            "particles": particles,
            "springs": springs,
        }


class SpringMeshAllPerturb(InitialConditionSource):
    def __init__(self, mesh_generator, magnitude_range=(0, 0.75)):
        super().__init__()
        self.mesh_generator = mesh_generator
        self.magnitude_range = magnitude_range
        self.n_dim = mesh_generator.n_dim
        self.n_particles = mesh_generator.n_particles
        assert self.n_dim == 2

    def particle_properties(self):
        particles, _springs = self.mesh_generator.generate_mesh()
        return particles

    def _sample_ring_uniform(self, inner_r, outer_r, num_pts=1):
        theta = np.random.uniform(0, 2*np.pi, num_pts)
        unifs = np.random.uniform(size=num_pts)
        r = np.sqrt(unifs * (outer_r**2 - inner_r**2) + inner_r**2)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack((x, y), axis=-1)

    def _generate_initial_condition(self):
        particles, springs = self.mesh_generator.generate_mesh()
        perturbs = self._sample_ring_uniform(min(self.magnitude_range),
                                             max(self.magnitude_range),
                                             num_pts=self.n_particles)
        # Apply perturbation
        for pi, particle in enumerate(particles):
            if particle["is_fixed"]:
                # Do not perturb fixed particles
                continue
            # Perturb this particle using interpolated values
            perturb = perturbs[pi, :]
            perturb_x = perturb[0]
            perturb_y = perturb[1]
            # Apply perturbation
            particle["position"][0] += perturb_x
            particle["position"][1] += perturb_y
        return {
            "particles": particles,
            "springs": springs,
        }


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


class NavierStokesInitialConditionSource(InitialConditionSource):
    def __init__(self,
                 velocity_range=(1.25, 1.75)):
        super().__init__()
        self.velocity_range = velocity_range

    def _generate_initial_condition(self):
        velocity = np.random.uniform(*self.velocity_range)
        return {
            "in_velocity": velocity,
        }


class NavierStokesMeshInitialConditionSource(NavierStokesInitialConditionSource):
    def __init__(
            self,
            velocity_range=(1.5, 1.5),
            edge_margin=(0.25, 0.05),
            radius_range=(0.05, 0.1),
            n_obstacles=1,
            pack_margin=0.05,
    ):
        super().__init__(velocity_range=velocity_range)
        self.velocity_range = velocity_range
        self.edge_margin = edge_margin
        self.radius_range = radius_range
        self.n_obstacles = n_obstacles
        self.pack_margin = pack_margin

    def _generate_initial_condition(self):
        template = super()._generate_initial_condition()
        meshes = []
        # Update with mesh components
        for _obstacle in range(self.n_obstacles):
            for _attempt in range(100):
                radius = np.random.uniform(*self.radius_range)
                # Select coordinates
                x_range = (0 + self.edge_margin[0] + radius, 2.2 - self.edge_margin[0] - radius)
                y_range = (0 + self.edge_margin[1] + radius, 0.41 - self.edge_margin[1] - radius)
                assert all(map(lambda x: 0 <= x <= 2.2, x_range))
                assert all(map(lambda y: 0 <= y <= 0.41, y_range))
                x = np.random.uniform(*x_range)
                y = np.random.uniform(*y_range)
                # Check that obstacle is suitable
                suitable = True
                for other_mesh in meshes:
                    o_x, o_y = other_mesh["center"]
                    o_r = other_mesh["radius"]
                    dist = np.sqrt((x - o_x)**2 + (y - o_y)**2)
                    if dist < o_r + radius + self.pack_margin:
                        suitable = False
                        break
                if suitable:
                    break
            else:
                raise ValueError("Failed to generate suitable obstacles")
            meshes.append({
                "radius": radius,
                "center": (x, y),
            })
        # Update template with mesh coordinates
        template.update({
            "mesh": meshes,
        })
        return template


class NavierStokesFixedInitialConditionSource(NavierStokesInitialConditionSource):
    def __init__(self,
                 fixed_velocities=None,
                 fallback_velocity_range=(1.25, 1.75)):
        super().__init__(velocity_range=fallback_velocity_range)
        self._conditions = (fixed_velocities or []).copy()
        self._conditions.reverse()

    def _generate_initial_condition(self):
        if self._conditions:
            return {
                "in_velocity": self._conditions.pop(),
            }
        else:
            return super()._generate_initial_condition()

class WritableDescription:
    def __init__(self, experiment, phase, name):
        self.experiment = experiment
        self.name = name
        self.phase = phase
        self.name_tag = None
        self._full_name = None

    def description(self):
        # Subclasses provide top-level dictionary including slurm_args
        # So dictionary with "phase_args" and "slurm_args" keys
        # Rest is filled in here
        raise NotImplementedError("Subclass this")

    @property
    def full_name(self):
        if self._full_name is None:
            self._full_name = self.experiment.get_run_name(self.name, name_tag=self.name_tag)
        return self._full_name

    @property
    def path(self):
        full_name = self.full_name
        return f"run/{self.phase}/{full_name}"

    @property
    def _descr_path(self):
        full_name = self.full_name
        return f"descr/{self.phase}/{full_name}.json"

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
        if(out_path.is_file()):
            raise ValueError(f"Description already exists at: {out_path}")
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
                 num_time_steps=30, time_step_size=0.3, subsampling=1,
                 noise_sigma=0,
                 mesh_based=False):
        super().__init__(experiment=experiment,
                         name_tail=f"n{num_traj}-t{num_time_steps}-n{noise_sigma}",
                         system="spring",
                         set_type=set_type)
        self.num_traj = num_traj
        self.initial_cond_source = initial_cond_source
        self.num_time_steps = num_time_steps
        self.time_step_size = time_step_size
        self.subsampling = subsampling
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
                "subsample": self.subsampling,
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
                 subsampling=1000, noise_sigma=0):
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



class SpringMeshDataset(Dataset):
    def __init__(self, experiment, initial_cond_source, num_traj,
                 set_type="train",
                 num_time_steps=500, time_step_size=0.1,
                 subsampling=10, noise_sigma=0, vel_decay=0.1):
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

        self.spatial_reshape = self.initial_cond_source.mesh_generator.grid_shape

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
                "time": "10:00:00",
                "cpus": min(16, len(trajectories)),
                "mem": 40,
            },
        }
        return template

    def input_size(self):
        return 2 * self.n_dim * self.n_particles


class NavierStokesDataset(Dataset):
    def __init__(self, experiment, initial_cond_source, num_traj,
                 set_type="train",
                 num_time_steps=10, time_step_size=0.08, subsampling=1):
        noise_sigma = 0
        super().__init__(experiment=experiment,
                         name_tail=f"n{num_traj}-t{num_time_steps}-n{noise_sigma}",
                         system="navier-stokes",
                         set_type=set_type)
        self.num_traj = num_traj
        self.initial_cond_source = initial_cond_source
        self.num_time_steps = num_time_steps
        self.time_step_size = time_step_size
        self.initial_conditions = self.initial_cond_source.sample_initial_conditions(self.num_traj)
        self.subsampling = subsampling
        assert isinstance(self.initial_cond_source, NavierStokesInitialConditionSource)
        self.spatial_reshape = (221, 42)

    def description(self):
        trajectories = []
        for icond in self.initial_conditions:
            traj = {
                "viscosity": 0.001,
                "in_velocity": 1.5,
                "num_time_steps": self.num_time_steps,
                "time_step_size": self.time_step_size,
                "subsample": self.subsampling,
            }
            traj.update(icond)
            trajectories.append(traj)
        # Generate template
        template = {
            "phase_args": {
                "system": "navier-stokes",
                "system_args": {
                    "grid_resolution": 0.01,
                    "trajectory_defs": trajectories,
                }
            },
            "slurm_args": {
                "gpu": False,
                "time": "08:00:00",
                "cpus": min(24, len(trajectories)),
                "mem": 40,
            },
        }
        return template

    def input_size(self):
        return 9282 * 3


class TrainedNetwork(WritableDescription):
    def __init__(
            self, experiment, method, name_tail,
            training_set, validation_set=None,
            gpu=True,
            learning_rate=1e-3, optimizer="adam", epochs=1000,
            train_dtype="float", batch_size=750, train_type=None,
            noise_variance=0,
            scheduler="none", scheduler_step="epoch", scheduler_args=None,
            predict_type="deriv",
            step_time_skew=1, step_subsample=1,
            flatten_input_data=True,
    ):
        super().__init__(experiment=experiment,
                         phase="train",
                         name=f"{method}-{name_tail}")
        self.method = method
        self.eval_gpu = True
        self.training_set = training_set
        self.validation_set = validation_set
        self.gpu = gpu
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_dtype = train_dtype
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.scheduler_step = scheduler_step
        self.scheduler_args = (scheduler_args or {}).copy()
        self.predict_type = predict_type
        self.step_time_skew = step_time_skew
        self.step_subsample = step_subsample
        self.train_type = train_type
        self.flatten_input_data = flatten_input_data
        self.noise_variance = noise_variance

        assert self.predict_type in {None, "step", "deriv"}

        if self.noise_variance == 0:
            self.noise_type = "none"
        elif self.predict_type == "step":
            self.noise_type = "step-corrected"
        elif self.predict_type == "deriv":
            self.noise_type = "deriv-corrected"

        # Check that the validation set is ok
        self.__check_val_set()

    def __check_val_set(self):
        if self.validation_set is None:
            return
        assert self.validation_set.system == self.training_set.system
        assert self.validation_set.input_size() == self.training_set.input_size()

    def description(self):
        template = {
            "phase_args": {
                "network": self.get_network_description(),
                "training": self.get_training_description(),
                "train_data": self.get_data_description(),
            },
            "slurm_args": self.get_slurm_args(),
        }
        return template

    def get_mem_requirement(self):
        system = self.training_set.system
        if system == "wave":
            if self.training_set.num_traj > 100 and self.training_set.num_time_steps > 8500:
                return 73
            else:
                return 37
        elif system == "navier-stokes":
            return 40
        elif system == "spring-mesh":
            return 40
        return 35

    def get_cpu_requirement(self):
        if self.gpu:
            return 4
        else:
            return 20

    def get_time_requirement(self):
        return "15:00:00"

    def get_network_description(self):
        raise NotImplementedError("Subclass this")

    def get_training_description(self):
        template = {
            "optimizer": self.optimizer,
            "optimizer_args": {
                "learning_rate": self.learning_rate,
            },
            "max_epochs": self.epochs,
            "try_gpu": self.gpu,
            "train_dtype": self.train_dtype,
            "train_type": self.train_type,
            "train_type_args": {},
            "scheduler": self.scheduler,
            "scheduler_step": self.scheduler_step,
            "scheduler_args": self.scheduler_args,
        }
        if self.noise_type and self.noise_type != "none":
            template["noise"] = {
                "type": self.noise_type,
                "variance": self.noise_variance,
            }
        return template

    def get_data_description(self):
        dataset_type = "snapshot"
        if self.predict_type == "step":
            dataset_type = "step-snapshot"
        template = {
            "data_dir": self.training_set.path,
            "dataset": dataset_type,
            "linearize": self.flatten_input_data,
            "dataset_args": {},
            "loader": {
                "batch_size": self.batch_size,
                "shuffle": True,
            }
        }
        if self.predict_type is not None:
            template["predict_type"] = self.predict_type
        if self.validation_set is not None:
            template["val_data_dir"] = self.validation_set.path
        if self.predict_type == "step":
            template["dataset_args"].update({
                "time-skew": self.step_time_skew,
                "subsample": self.step_subsample,
            })
        return template

    def get_slurm_args(self):
        return {
            "gpu": self.gpu,
            "time": self.get_time_requirement(),
            "cpus": self.get_cpu_requirement(),
            "mem": self.get_mem_requirement(),
        }


class MLP(TrainedNetwork):
    def __init__(self, experiment, training_set, gpu=True, learning_rate=1e-3,
                 hidden_dim=2048, depth=2, train_dtype="float",
                 scheduler="none", scheduler_step="epoch",
                 batch_size=750, epochs=1000, validation_set=None,
                 noise_variance=0, predict_type="deriv",
                 step_time_skew=1, step_subsample=1):
        super().__init__(
            experiment=experiment,
            method=f"mlp-{predict_type}",
            name_tail=f"{training_set.name}-d{depth}-h{hidden_dim}",
            training_set=training_set,
            validation_set=validation_set,
            gpu=gpu,
            learning_rate=learning_rate,
            optimizer="adam",
            epochs=epochs,
            train_dtype=train_dtype,
            batch_size=batch_size,
            train_type=f"mlp-{predict_type}",
            noise_variance=noise_variance,
            scheduler=scheduler,
            scheduler_step=scheduler_step,
            scheduler_args=None,
            predict_type=predict_type,
            step_time_skew=step_time_skew,
            step_subsample=step_subsample,
            flatten_input_data=True,
        )
        self.hidden_dim = hidden_dim
        self.depth = depth

        assert self.predict_type in {"step", "deriv"}

        self.input_size = self.training_set.input_size()
        self.output_size = self.training_set.input_size()
        if self.training_set.system == "navier-stokes":
            # Extra data support for Navier-Stokes
            extra_dims = 2 * (self.training_set.spatial_reshape[0] * self.training_set.spatial_reshape[1])
            self.input_size += extra_dims
        elif self.training_set.system == "spring-mesh":
            # Extra data support for Navier-Stokes
            extra_dims = self.training_set.spatial_reshape[0] * self.training_set.spatial_reshape[1]
            self.input_size += extra_dims

    def get_network_description(self):
        template = {
            "arch": f"mlp-{self.predict_type}",
            "arch_args": {
                "input_dim": self.input_size,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_size,
                "depth": self.depth,
                "nonlinearity": "tanh",
            },
        }
        return template


class CNN(TrainedNetwork):
    def __init__(self, experiment, training_set, gpu=True, learning_rate=1e-3,
                 chans_inout_kenel=((None, 32, 5), (32, None, 5)),
                 train_dtype="float",
                 scheduler="none", scheduler_step="epoch", scheduler_args={},
                 batch_size=750, epochs=1000, validation_set=None,
                 noise_variance=0, predict_type="deriv",
                 padding_mode="zeros",
                 step_time_skew=1, step_subsample=1):
        if training_set.system == "spring-mesh":
            base_num_chans = 5
        elif training_set.system in {"navier-stokes"}:
            base_num_chans = 5
        elif training_set.system == "wave":
            base_num_chans = 2
        chan_records = [
            {
                "kernel_size": ks,
                "in_chans": ic or base_num_chans,
                "out_chans": oc or base_num_chans,
            }
            for ic, oc, ks in chans_inout_kenel]
        name_key = ";".join([f"{cr['kernel_size']}:{cr['in_chans']}" for cr in chan_records])
        super().__init__(
            experiment=experiment,
            method=f"cnn-{predict_type}",
            name_tail=f"{training_set.name}-a{name_key}",
            training_set=training_set,
            validation_set=validation_set,
            gpu=gpu,
            learning_rate=learning_rate,
            optimizer="adam",
            epochs=epochs,
            train_dtype=train_dtype,
            batch_size=batch_size,
            train_type=f"cnn-{predict_type}",
            noise_variance=noise_variance,
            scheduler=scheduler,
            scheduler_step=scheduler_step,
            scheduler_args=None,
            predict_type=predict_type,
            step_time_skew=step_time_skew,
            step_subsample=step_subsample,
            flatten_input_data=False,
        )
        self.layer_defs = chan_records
        self.padding_mode = padding_mode

        assert self.predict_type in {"step", "deriv"}

        self.conv_dim = 1
        if training_set.system in {"navier-stokes", "spring-mesh"}:
           self.conv_dim = 2
        self.spatial_reshape = getattr(self.training_set, "spatial_reshape", None)

    def get_network_description(self):
        template = {
            "arch": f"cnn-{self.predict_type}",
            "arch_args": {
                "nonlinearity": "relu",
                "layer_defs": self.layer_defs,
                "dim": self.conv_dim,
                "spatial_reshape": self.spatial_reshape,
                "padding_mode": self.padding_mode,
            },
        }
        return template


class UNet(TrainedNetwork):
    def __init__(self, experiment, training_set, gpu=True, learning_rate=1e-3,
                 train_dtype="float",
                 scheduler="none", scheduler_step="epoch", scheduler_args={},
                 batch_size=750, epochs=1000, validation_set=None,
                 noise_variance=0, predict_type="deriv",
                 loss_type="l1",
                 step_time_skew=1, step_subsample=1):
        super().__init__(
            experiment=experiment,
            method=f"unet-{predict_type}",
            name_tail=f"{training_set.name}",
            training_set=training_set,
            validation_set=validation_set,
            gpu=gpu,
            learning_rate=learning_rate,
            optimizer="adam",
            epochs=epochs,
            train_dtype=train_dtype,
            batch_size=batch_size,
            train_type=f"unet-{predict_type}",
            noise_variance=noise_variance,
            scheduler=scheduler,
            scheduler_step=scheduler_step,
            scheduler_args=None,
            predict_type=predict_type,
            step_time_skew=step_time_skew,
            step_subsample=step_subsample,
            flatten_input_data=False,
        )

        assert self.predict_type in {"step", "deriv"}

        self._predict_system = training_set.system
        self.loss_type = loss_type
        assert self._predict_system in {"navier-stokes", "spring-mesh"}
        self.spatial_reshape = getattr(self.training_set, "spatial_reshape", None)

    def get_network_description(self):
        template = {
            "arch": f"unet-{self.predict_type}",
            "arch_args": {
                "predict_system": self._predict_system,
                "spatial_reshape": self.spatial_reshape,
            },
        }
        return template

    def get_training_description(self):
        template = super().get_training_description()
        template["loss_type"] = self.loss_type
        return template


class NNKernel(TrainedNetwork):
    def __init__(self, experiment, training_set, gpu=True, learning_rate=1e-3,
                 hidden_dim=2048, train_dtype="float",
                 batch_size=750, epochs=1000, validation_set=None,
                 nonlinearity="relu", optimizer="sgd", weight_decay=0,
                 predict_type="deriv",
                 step_time_skew=1, step_subsample=1):
        super().__init__(
            experiment=experiment,
            method=f"nn-kernel-{predict_type}",
            name_tail=f"{training_set.name}-h{hidden_dim}-lr{learning_rate}-wd{weight_decay}",
            training_set=training_set,
            validation_set=validation_set,
            gpu=gpu,
            learning_rate=learning_rate,
            optimizer=optimizer,
            epochs=epochs,
            train_dtype=train_dtype,
            batch_size=batch_size,
            train_type=f"nn-kernel-{predict_type}",
            noise_variance=0,
            scheduler="none",
            predict_type=predict_type,
            step_time_skew=step_time_skew,
            step_subsample=step_subsample,
            flatten_input_data=True,
        )

        assert self.predict_type in {"step", "deriv"}

        self.hidden_dim = hidden_dim
        self.nonlinearity = nonlinearity
        self.weight_decay = weight_decay

    def get_network_description(self):
        template = {
            "arch": f"nn-kernel-{self.predict_type}",
            "arch_args": {
                "input_dim": self.training_set.input_size(),
                "hidden_dim": self.hidden_dim,
                "output_dim": self.training_set.input_size(),
                "nonlinearity": self.nonlinearity,
            },
        }
        return template

    def get_training_description(self):
        template = super().get_training_description()
        template["optimizer_args"]["weight_decay"] = self.weight_decay
        return template

    def get_time_requirement(self):
        return "10:00:00"


class Evaluation(WritableDescription):
    def __init__(self, experiment, name_tail):
        super().__init__(experiment=experiment,
                         phase="eval",
                         name=f"eval-{name_tail}")

    def _get_mem_requirement(self, eval_set):
        system = eval_set.system
        return 32


class NetworkEvaluation(Evaluation):
    def __init__(self, experiment, network, eval_set, gpu=None, integrator=None,
                 eval_dtype=None, network_file="model.pt", time_limit=None):
        if network.method in {"knn-predictor", "knn-predictor-oneshot", "cnn-step", "mlp-step", "nn-kernel-step", "unet-step"}:
            integrator = "null"
        if integrator is None:
            raise ValueError("Must manually specify integrator")
        super().__init__(experiment=experiment,
                         name_tail=f"net-{network.name}-set-{eval_set.name}-{integrator}")
        self._time_limit = time_limit or "01:30:00"
        self.network = network
        self.network_file = network_file
        self.eval_set = eval_set
        if gpu is None:
            # Auto handling
            self.gpu = network.eval_gpu
        else:
            self.gpu = gpu
        if eval_dtype is None:
            self.eval_dtype = self.network.train_dtype
        else:
            self.eval_dtype = eval_dtype
        self.integrator = integrator
        # Validate inputs
        if self.eval_set.system != self.network.training_set.system:
            raise ValueError(f"Inconsistent systems {self.eval_set.system} and {self.network.training_set.system}")


    def description(self):
        eval_type = self.network.method
        gpu = self.gpu and (eval_type not in {"knn-predictor", "knn-regressor", "knn-predictor-oneshot", "knn-regressor-oneshot"})
        template = {
            "phase_args": {
                "eval_net": self.network.path,
                "eval_net_file": self.network_file,
                "eval_data": {
                    "data_dir": self.eval_set.path,
                    "linearize": (self.network.method not in {"cnn", "cnn-step", "cnn-deriv", "unet-step", "unet-deriv"}),
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
                "time": self._time_limit,
                "cpus": 4 if gpu else 20,
                "mem": self._get_mem_requirement(eval_set=self.eval_set),
            },
        }
        return template


class KNNOneshotEvaluation(NetworkEvaluation):
    MockNetwork = dataclasses.make_dataclass("MockNetwork", ["name", "train_dtype", "training_set", "method", "path"])

    def __init__(self, experiment, training_set, eval_set, knn_type,
                 eval_dtype="double", integrator="leapfrog",
                 dataset_type="snapshot", dataset_args=None, batch_size=750):
        method = f"knn-{knn_type}-oneshot"
        self._mock_network = self.MockNetwork(name=f"{method}-{training_set.name}",
                                              train_dtype=eval_dtype,
                                              training_set=training_set,
                                              method=method,
                                              path=None)
        super().__init__(experiment=experiment, network=self._mock_network,
                         eval_set=eval_set, gpu=False, integrator=integrator,
                         eval_dtype=eval_dtype, time_limit="16:00:00")
        self.training_set = training_set
        self.dataset_type = dataset_type
        self.dataset_args = dataset_args or {}
        self.batch_size = batch_size

    def description(self):
        template = super().description()
        # Add arguments to construct the training set
        template["phase_args"]["eval"]["train_data"] = {
            "data_dir": self.training_set.path,
            "dataset": self.dataset_type,
            "linearize": True,
            "dataset_args": {},
            "loader": {
                "batch_size": self.batch_size,
                "shuffle": False,
            },
        }
        template["phase_args"]["eval"]["train_data"]["dataset_args"].update(self.dataset_args)
        return template


class KNNPredictorOneshot(KNNOneshotEvaluation):
    def __init__(self, experiment, training_set, eval_set,
                 eval_dtype="double", step_time_skew=1, step_subsample=1):
        super().__init__(experiment=experiment,
                         training_set=training_set,
                         eval_set=eval_set,
                         eval_dtype=eval_dtype,
                         integrator="null",
                         knn_type="predictor",
                         batch_size=1,
                         dataset_type="step-snapshot",
                         dataset_args={
                             "time-skew": step_time_skew,
                             "subsample": step_subsample,
                         })


class KNNRegressorOneshot(KNNOneshotEvaluation):
    def __init__(self, experiment, training_set, eval_set,
                 eval_dtype="double", integrator="leapfrog"):
        super().__init__(experiment=experiment,
                         training_set=training_set,
                         eval_set=eval_set,
                         eval_dtype=eval_dtype,
                         integrator=integrator,
                         dataset_type="snapshot",
                         knn_type="regressor")


class BaselineIntegrator(Evaluation):
    def __init__(self, experiment, eval_set, integrator="leapfrog",
                 eval_dtype="double", coarsening=1):
        super().__init__(experiment=experiment,
                         name_tail=f"integrator-baseline-{eval_set.name}-{integrator}-{eval_dtype}")
        self.eval_set = eval_set
        self.eval_dtype = eval_dtype
        self.integrator = integrator
        self.coarsening = coarsening

    def description(self):
        template = {
            "phase_args": {
                "eval_net": None,
                "eval_data": {
                    "data_dir": self.eval_set.path,
                    "linearize": False,
                },
                "eval": {
                    "eval_type": "integrator-baseline",
                    "integrator": self.integrator,
                    "eval_dtype": self.eval_dtype,
                    "try_gpu": False,
                    "coarsening": self.coarsening,
                },
            },
            "slurm_args": {
                "gpu": False,
                "time": "04:00:00",
                "cpus": 16,
                "mem": self._get_mem_requirement(eval_set=self.eval_set),
            },
        }
        if self.eval_set.system == "spring-mesh":
            template["phase_args"]["eval_data"]["linearize"] = True
        return template
