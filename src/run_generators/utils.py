import copy
import numpy as np
import pathlib
import json
import math
import re
import dataclasses
import itertools
from scipy import interpolate


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
    elif system == "taylor-green":
        dim = dataset.input_size() // 2
        instance.particle_process_type = "identity"
        instance.adjacency_args = {
            "type": "native",
            "boundary_conditions": "periodic",
            "boundary_vertices": None,
            "dimension": dim,
            }
        instance.v_features = [4, 5]
        instance.e_features = 6
        instance.mesh_coords = [list(map(float, p)) for p in dataset.initial_cond_source.vertices]
        instance.static_nodes = [1 for p in dataset.initial_cond_source.vertices]
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


class TaylorGreenGridGenerator:
    def __init__(self, grid_shape):
        self.grid_shape = grid_shape
        self.n_grid = self.grid_shape[0]
        assert all(v == self.grid_shape[0] for v in self.grid_shape)
        self.n_dims = len(grid_shape)
        self.n_dim = self.n_dims
        self.n_vertices = 1
        for s in grid_shape:
            self.n_vertices *= s
        self._vertices = None
        self._edges = None

    def generate_mesh(self):
        if self._vertices is None:
            vertices = []
            edges = []
            ranges = [range(s) for s in self.grid_shape]
            # Generate vertices
            for coords in itertools.product(*ranges):
                vertices.append(list(coords))
            # Add edges
            for (a, part_a), (b, part_b) in itertools.combinations(enumerate(vertices), 2):
                # Determine if we want an edge
                length = math.sqrt(sum((pa - pb) ** 2 for pa, pb in zip(part_a, part_b)))
                periodic_boundary_x = (part_a[1] == part_b[1] and np.allclose(abs(part_a[0] - part_b[0]), self.grid_shape[0]))
                periodic_boundary_y = (part_a[0] == part_b[0] and np.allclose(abs(part_a[1] - part_b[1]), self.grid_shape[1]))
                if not (np.allclose(length, 1) or periodic_boundary_x or periodic_boundary_y):
                    continue
                # Add the edge
                edge_def = {
                    "a": a,
                    "b": b,
                }
                edges.append(edge_def)
            self._vertices = vertices
            self._edges = edges
        return copy.deepcopy(self._vertices), copy.deepcopy(self._edges)


class TaylorGreenInitialConditionSource(InitialConditionSource):
    def __init__(self,
                 mesh_generator,
                 viscosity_range=(0.5, 1.5),
                 density_range=(1.0, 1.0)):
        super().__init__()
        self.viscosity_range = viscosity_range
        self.density_range = density_range
        self.vertices, self.edges = mesh_generator.generate_mesh()
        self.mesh_generator = mesh_generator

    def _generate_initial_condition(self):
        viscosity = np.random.uniform(*self.viscosity_range)
        density = np.random.uniform(*self.density_range)
        state = {
            "viscosity": viscosity,
            "density": density,
            "vertices": self.vertices,
            "edges": self.edges,
        }
        return state


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


class ParticleDataset(Dataset):
    def __init__(self, experiment, initial_cond_source, num_traj,
                 set_type="train", n_dim=2, n_particles=2,
                 num_time_steps=200, time_step_size=0.1, noise_sigma=0,
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
                "time": "24:00:00",
                "cpus": 4,
                "mem": 40,
            },
        }
        return template

    def input_size(self):
        return 2 * self.n_dim * self.n_particles


class TaylorGreenDataset(Dataset):
    def __init__(self, experiment, initial_cond_source, num_traj,
                 set_type="train", n_grid=250,
                 num_time_steps=200, time_step_size=0.1):
        noise_sigma = 0
        super().__init__(experiment=experiment,
                         name_tail=f"n{num_traj}-t{num_time_steps}-n{noise_sigma}",
                         system="taylor-green",
                         set_type=set_type)
        self.n_grid = n_grid
        self.num_traj = num_traj
        self.initial_cond_source = initial_cond_source
        self.num_time_steps = num_time_steps
        self.time_step_size = time_step_size
        self.initial_conditions = self.initial_cond_source.sample_initial_conditions(self.num_traj)
        assert isinstance(self.initial_cond_source, TaylorGreenInitialConditionSource)
        assert self.n_grid == self.initial_cond_source.mesh_generator.n_grid

    def description(self):
        trajectories = []
        for icond in self.initial_conditions:
            traj = {
                "viscosity": 1.0,
                "space_scale": 2,
                "density": 1.0,
                "num_time_steps": self.num_time_steps,
                "time_step_size": self.time_step_size,
            }
            traj.update(icond)
            trajectories.append(traj)
        # Generate template
        template = {
            "phase_args": {
                "system": "taylor-green",
                "system_args": {
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
        return 3 * (self.n_grid ** 2)


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
                "time": "05:00:00",
                "cpus": 4,
                "mem": 15,
            },
        }
        return template

    def input_size(self):
        return 9282 * 3


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
        elif system == "navier-stokes":
            return 32
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
                "time": "6:00:00",
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
                 noise_type="none", noise_variance=0, predict_type="deriv",
                 step_time_skew=1, step_subsample=1):
        super().__init__(experiment=experiment,
                         method="-".join(["mlp", predict_type]),
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
        self.predict_type = predict_type
        self.step_time_skew = step_time_skew
        self.step_subsample = step_subsample
        generate_scheduler_args(self, end_lr)
        assert predict_type in {"deriv", "step"}

    def description(self):
        dataset_type = "snapshot"
        if self.training_set == "taylor-green":
            dataset_type = "taylor-green"
        elif self.predict_type == "step":
            dataset_type = "step-snapshot"
        template = {
            "phase_args": {
                "network": {
                    "arch": "-".join(["mlp", self.predict_type]),
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
                    "train_type": "-".join(["mlp", self.predict_type]),
                    "train_type_args": {},
                    "scheduler": self.scheduler,
                    "scheduler_step": self.scheduler_step,
                    "scheduler_args": self.scheduler_args,
                },
                "train_data": {
                    "data_dir": self.training_set.path,
                    "dataset": dataset_type,
                    "predict_type": self.predict_type,
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
                "time": "6:00:00",
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
        if self.predict_type == "step":
             template["phase_args"]["train_data"]["dataset_args"].update({
                 "time-skew": self.step_time_skew,
                 "subsample": self.step_subsample,
             })
        return template


class CNN(TrainedNetwork):
    def __init__(self, experiment, training_set, gpu=True, learning_rate=1e-3,
                 chans_inout_kenel=((None, 32, 5), (32, None, 5)),
                 train_dtype="float",
                 scheduler="none", scheduler_step="epoch", scheduler_args={},
                 batch_size=750, epochs=1000, validation_set=None,
                 noise_type="none", noise_variance=0, predict_type="deriv"):
        base_num_chans = getattr(training_set, "n_particles", 2)
        chan_records = [
            {
                "kernel_size": ks,
                "in_chans": ic or base_num_chans,
                "out_chans": oc or base_num_chans,
            }
            for ic, oc, ks in chans_inout_kenel]
        name_key = ";".join([f"{cr['kernel_size']}:{cr['in_chans']}:{cr['out_chans']}" for cr in chan_records])
        super().__init__(experiment=experiment,
                         method="-".join(["cnn", predict_type]),
                         name_tail=f"{training_set.name}-a{name_key}")
        self.training_set = training_set
        self.gpu = gpu
        self.learning_rate = learning_rate
        self.train_dtype = train_dtype
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_set = validation_set
        self.scheduler = scheduler
        self.scheduler_step = scheduler_step
        self.scheduler_args = {}
        self._check_val_set(train_set=self.training_set, val_set=self.validation_set)
        self.noise_type = noise_type
        self.noise_variance = noise_variance
        self.layer_defs = chan_records
        self.predict_type = predict_type
        assert predict_type in {"deriv", "step"}

    def description(self):
        template = {
            "phase_args": {
                "network": {
                    "arch": "-".join(["cnn", self.predict_type]),
                    "arch_args": {
                        "nonlinearity": "relu",
                        "layer_defs": self.layer_defs,
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
                    "train_type": "-".join(["cnn", self.predict_type]),
                    "train_type_args": {},
                    "scheduler": self.scheduler,
                    "scheduler_step": self.scheduler_step,
                    "scheduler_args": self.scheduler_args,
                },
                "train_data": {
                    "data_dir": self.training_set.path,
                    "dataset": ("snapshot" if self.predict_type == "deriv" else "step-snapshot"),
                    "predict_type": self.predict_type,
                    "linearize": False,
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
                 nonlinearity="relu", optimizer="sgd", weight_decay=0,
                 predict_type="deriv"):
        super().__init__(experiment=experiment,
                         method="-".join(["nn-kernel", predict_type]),
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
        self.predict_type = predict_type
        assert predict_type in {"deriv", "step"}

    def description(self):
        dataset_type = "snapshot"
        if self.training_set == "taylor-green":
            dataset_type = "taylor-green"
        elif self.predict_type == "step":
            dataset_type = "step-snapshot"
        template = {
            "phase_args": {
                "network": {
                    "arch": "-".join(["nn-kernel", self.predict_type]),
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
                    "train_type": "-".join(["nn-kernel", self.predict_type]),
                    "train_type_args": {},
                },
                "train_data": {
                    "data_dir": self.training_set.path,
                    "dataset": dataset_type,
                    "predict_type": self.predict_type,
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
                "time": "6:00:00",
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
                    "linearize": (self.network.method not in {"hogn", "gn", "cnn"}),
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
                "cpus": 4 if gpu else 20,
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
                         eval_dtype=eval_dtype)
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
                         dataset_type=("taylor-green" if training_set.system == "taylor-green" else "snapshot"),
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
                    "linearize": (self.eval_set.system == "taylor-green"),
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
        if self.eval_set.system == "spring-mesh":
            template["phase_args"]["eval_data"]["linearize"] = True
        return template
