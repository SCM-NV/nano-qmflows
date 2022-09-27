"""Dataclasses used for storing nanoqm settings."""

from __future__ import annotations

import os
import sys
import textwrap
import pprint
import functools
import dataclasses
import warnings
from typing import Any, TYPE_CHECKING, Protocol, Literal

import qmflows
from scm.plams import Settings
from qmflows.common import AtomXYZ
from qmflows.warnings_qmflows import QMFlowsDeprecationWarning

if TYPE_CHECKING:
    from dataclasses import dataclass
    from numpy.typing import NDArray
    from numpy import float32 as f4, float64 as f8
    from .schedule.components import JobFiles

    class _FunCoupling(Protocol):
        def __call__(self, dt: float, *args: NDArray[f8]) -> NDArray[f8]:
            ...

else:
    if sys.version_info >= (3, 10):
        dataclass = functools.partial(dataclasses.dataclass, repr=False, kw_only=True, slots=True)
    else:
        dataclass = functools.partial(dataclasses.dataclass, repr=False)

pformat = functools.partial(pprint.pformat, compact=True, sort_dicts=False)

__all__ = [
    "CP2KGeneralSetting",
    "JobScheduler",
    "GeneralOptions",
    "SinglePoints",
    "AbsorptionSpectrum",
    "DerivativeCoupling",
    "Distribute",
    "DistributeDerivativeCoupling",
    "DistributeAbsorptionSpectrum",
    "DistributeSinglePoints",
    "IPR",
    "COOP",
    "AbsorptionData",
    "DistributeData",
    "ComponentsData",
]


@dataclass
class _DataConfig:
    """Nanoqm dataclass baseclass with a number of preset methods."""

    def __repr__(self) -> str:
        """Implement ``repr(self)``."""
        cls = type(self)
        data = ""
        field_iter = ((field.name, getattr(self, field.name)) for field in dataclasses.fields(self))
        for name, field in field_iter:
            width = 94 - len(name)
            offset = 100 - width
            data += f"    {name}: "
            data += textwrap.indent(pformat(field, width=width), offset * " ")[offset:]
            data += ",\n"
        return f"{cls.__name__}(\n{data})"

    def asdict(self) -> dict[str, Any]:
        """Convert this instance into a yaml-safe dictionary."""
        ret = dataclasses.asdict(self)
        remove = set()
        for k, v in ret.items():
            if v is NotImplemented:
                remove.add(k)
            else:
                ret[k] = self._recursive_traverse(v)
        for k in remove:
            del ret[k]
        return ret

    @classmethod
    def _recursive_traverse(cls, val: object) -> Any:
        """Check if the value of a key is a Settings instance a transform it to plain dict."""
        if isinstance(val, dict):
            if isinstance(val, Settings):
                return val.as_dict()
            else:
                return {k: cls._recursive_traverse(v) for k, v in val.items()}
        elif isinstance(val, tuple):
            return list(val)
        elif isinstance(val, _DataConfig):
            return cls._recursive_traverse(val.asdict())
        else:
            return val

    # There should be no instances of get-/setitem left in `nanoqm`,
    # but there might be a few downstream
    if not TYPE_CHECKING:
        def __getitem__(self, key: str) -> Any:
            """Implement ``self[key]``."""
            ret = getattr(self, key)
            warnings.warn(
                f"`self[{key!r}]` attribute getting has been deprecated and will be removed "
                f"in the future; use `self.{key}` instead",
                QMFlowsDeprecationWarning, stacklevel=2,
            )
            return ret

        def __setitem__(self, key: str, value: Any) -> None:
            """Implement ``self[key] = value``."""
            setattr(self, key, value)
            warnings.warn(
                f"`self[{key!r}] = ...` attribute setting has been deprecated and will be removed "
                f"in the future; use `self.{key} = ...` instead",
                QMFlowsDeprecationWarning, stacklevel=2,
            )


@dataclass
class CP2KGeneralSetting(_DataConfig):
    """Dataclass for CP2K general settings."""

    basis: str
    potential: str
    charge: int
    multiplicity: int
    cell_parameters: float | list[float] | list[list[float]]
    periodic: Literal["none", "x", "y", "z", "xy", "xy", "yz", "xyz"]
    cell_angles: None | list[float]
    path_basis: str
    basis_file_name: None | list[str]
    potential_file_name: None | str
    functional_x: None | str
    functional_c: None | str
    cp2k_settings_main: qmflows.Settings
    cp2k_settings_guess: qmflows.Settings
    wfn_restart_file_name: None | str
    file_cell_parameters: None | str
    aux_fit: Literal["low", "medium", "good", "verygood", "excellent"]
    executable: str


@dataclass
class JobScheduler(_DataConfig):
    """Dataclass with options for a job scheduler."""

    scheduler: Literal["slurm", "pbs"]
    nodes: int
    tasks: int
    wall_time: str
    job_name: str
    queue_name: str
    load_modules: str
    free_format: str


@dataclass
class GeneralOptions(_DataConfig):
    """Dataclass with options common to all workflows."""

    active_space: tuple[int, int]
    nHOMO: int
    mo_index_range: tuple[int, int]
    package_name: str
    project_name: str
    scratch_path: str | os.PathLike[str]
    path_hdf5: str
    path_traj_xyz: str
    enumerate_from: int
    ignore_warnings: bool
    calculate_guesses: Literal["first", "all"]
    geometry_units: Literal["angstrom", "au"]
    dt: float
    compute_orbitals: bool
    remove_log_file: bool
    cp2k_general_settings: CP2KGeneralSetting
    orbitals_type: Literal["", "alphas", "betas", "both"]

    if sys.version_info >= (3, 10):
        multiplicity: int = NotImplemented
        workdir: str | os.PathLike[str] = NotImplemented
        geometries: list[str] = NotImplemented
        folders: list[str] = NotImplemented
        calc_new_wf_guess_on_points: list[int] = NotImplemented
    else:
        def __post_init__(self) -> None:
            """Attach later to-be populated attribures."""
            self.multiplicity: int = NotImplemented
            self.geometries: list[str] = NotImplemented
            self.folders: list[str] = NotImplemented
            self.calc_new_wf_guess_on_points: list[int] = NotImplemented
            if not hasattr(self, "workdir"):
                self.workdir: str | os.PathLike[str] = NotImplemented


@dataclass
class SinglePoints(GeneralOptions):
    """Dataclass with options for a single point calculations."""

    workflow: Literal["single_points", "ipr_calculation", "coop_calculation"]


@dataclass
class AbsorptionSpectrum(GeneralOptions):
    """Dataclass with options for an absorption spectrum calculation."""

    workflow: Literal["absorption_spectrum"]
    tddft: Literal["sing_orb", "stda", "stdft"]
    stride: int
    xc_dft: str


@dataclass
class DerivativeCoupling(GeneralOptions):
    """Dataclass with options for a derivative coupling calculation."""

    workflow: Literal["derivative_couplings"]
    algorithm: Literal["levine", "3points"]
    mpi: bool
    tracking: bool
    write_overlaps: bool
    overlaps_deph: bool

    if sys.version_info >= (3, 10):
        fun_coupling: _FunCoupling = NotImplemented
        path_hamiltonians: str = NotImplemented
        npoints: int = NotImplemented
    else:
        def __post_init__(self) -> None:
            """Attach later to-be populated attribures."""
            super().__post_init__()
            self.fun_coupling: _FunCoupling = NotImplemented
            self.path_hamiltonians: str = NotImplemented
            self.npoints: int = NotImplemented


# Can't inherit from multiple baseclasses due to a __slots__ conflict (dataclass bug?),
# but we can pretend that `Distribute` is a baseclass
if TYPE_CHECKING:
    @dataclass
    class Distribute(GeneralOptions):
        """Type check only dataclass for distributed jobs."""

        workdir: str | os.PathLike[str]
        blocks: int
        job_scheduler: JobScheduler
        stride: None = None
else:
    Distribute = object


@dataclass
class DistributeDerivativeCoupling(DerivativeCoupling, Distribute):
    """Dataclass for distributing derivative coupling jobs."""

    workflow: Literal["distribute_derivative_couplings"]
    workdir: str | os.PathLike[str]
    blocks: int
    job_scheduler: JobScheduler
    stride: None = None


@dataclass
class DistributeAbsorptionSpectrum(AbsorptionSpectrum, Distribute):
    """Dataclass for distributing absorption spectrum jobs."""

    workflow: Literal["distribute_absorption_spectrum"]
    workdir: str | os.PathLike[str]
    blocks: int
    job_scheduler: JobScheduler


@dataclass
class DistributeSinglePoints(SinglePoints, Distribute):
    """Dataclass for distributing single point jobs."""

    workflow: Literal["distribute_single_points"]
    workdir: str | os.PathLike[str]
    blocks: int
    job_scheduler: JobScheduler
    stride: None = None


@dataclass
class IPR(SinglePoints):
    """Dataclass with options for an Inverse Participation Ratio calculation."""

    workflow: Literal["ipr_calculation"]


@dataclass
class COOP(SinglePoints):
    """Dataclass with options for a Crystal Orbital Overlap Population calculation."""

    workflow: Literal["coop_calculation"]
    coop_elements: tuple[str, str]


@dataclass
class AbsorptionData(_DataConfig):
    """Dataclass with data related to absorption spectrum workflows."""

    i: int
    mol: list[AtomXYZ]
    energy: NDArray[f4] = NotImplemented
    c_ao: NDArray[f4] = NotImplemented
    nocc: int = NotImplemented
    nvirt: int = NotImplemented
    overlap: NDArray[f8] = NotImplemented
    multipoles: NDArray[f8] = NotImplemented
    omega: NDArray[f8] = NotImplemented
    xia: NDArray[f8] = NotImplemented
    dipole: tuple[NDArray[f8], NDArray[f8], NDArray[f8]] = NotImplemented
    oscillator: NDArray[f8] = NotImplemented


@dataclass
class DistributeData(_DataConfig):
    """Dataclass with data related to distributed workflows."""

    folder_path: str
    file_xyz: str
    index: int
    hamiltonians_dir: None | str


@dataclass
class ComponentsData(_DataConfig):
    """Dataclass with data related to components."""

    geometry: str
    k: int
    node_MOs: tuple[str, str, str]
    node_energy: str
    point_dir: str = NotImplemented
    job_files: JobFiles = NotImplemented
    job_name: str = NotImplemented
