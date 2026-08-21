"""cadgenesis.agents.loader
=========================
Dynamic agent loader.

Loads :class:`~cadgenesis.agents.base.Agent` classes from installed modules,
filesystem plugin directories, or a configured module list.  Every loaded agent
is validated (subclass of ``Agent``, has a ``role``, version parses) before it
is offered to a registry.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path

from cadgenesis.agents.base import Agent
from cadgenesis.agents.infrastructure import AgentBase
from cadgenesis.agents.versioning import AgentVersion


class AgentLoadError(RuntimeError):
    """Raised when an agent cannot be discovered or loaded."""


class AgentLoader:
    """Discovers and instantiates agent classes from modules and packages."""

    def __init__(self, package: str = "cadgenesis.agents") -> None:
        self._package = package

    # --------------------------------------------------------------- scanning

    def scan_package(self, package_name: str, recursive: bool = True) -> list[Agent]:
        """Import and instantiate every :class:`Agent` subclass in ``package_name``."""
        try:
            package = importlib.import_module(package_name)
        except ImportError as exc:
            raise AgentLoadError(f"cannot import package {package_name!r}: {exc}") from exc
        agents: list[Agent] = []
        for module_info in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}."):
            if not recursive and module_info.ispkg:
                continue
            agents.extend(self.load_module(module_info.name))
        return agents

    def scan_directory(self, directory: str | Path, package_prefix: str = "") -> list[Agent]:
        """Load every ``*.py`` module (non-``__``) from a filesystem directory."""
        path = Path(directory)
        if not path.is_dir():
            raise AgentLoadError(f"plugin directory {path} does not exist")
        agents: list[Agent] = []
        for module_path in sorted(path.glob("*.py")):
            if module_path.name.startswith("__"):
                continue
            module_name = module_path.stem
            if package_prefix:
                module_name = f"{package_prefix}.{module_name}"
            agents.extend(self.load_module(module_name))
        return agents

    # ------------------------------------------------------------ module load

    def load_module(self, module_name: str) -> list[Agent]:
        """Import ``module_name`` and instantiate every Agent subclass it defines."""
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise AgentLoadError(f"cannot import module {module_name!r}: {exc}") from exc
        agents: list[Agent] = []
        for _, candidate in inspect.getmembers(module, inspect.isclass):
            if candidate is Agent or candidate is AgentBase:
                continue
            if not issubclass(candidate, Agent):
                continue
            if inspect.isabstract(candidate):
                continue
            agents.append(self._instantiate(candidate, module_name))
        return agents

    def load_class(self, qualified_name: str) -> Agent:
        """Instantiate the class at ``"package.module.ClassName"``."""
        module_name, _, class_name = qualified_name.rpartition(".")
        if not module_name:
            raise AgentLoadError(
                f"invalid qualified class name {qualified_name!r}: expected module.path.Class"
            )
        module = importlib.import_module(module_name)
        candidate = getattr(module, class_name, None)
        if candidate is None or not (inspect.isclass(candidate) and issubclass(candidate, Agent)):
            raise AgentLoadError(f"{qualified_name!r} is not an Agent class")
        return self._instantiate(candidate, module_name)

    # -------------------------------------------------------------- validation

    @staticmethod
    def _instantiate(candidate: type[Agent], source: str) -> Agent:
        try:
            agent = candidate()
        except Exception as exc:
            raise AgentLoadError(
                f"cannot instantiate {candidate.__name__} from {source}: {exc}"
            ) from exc
        if not getattr(agent, "role", ""):
            raise AgentLoadError(f"{candidate.__name__} from {source} defines no role")
        if isinstance(agent, AgentBase):
            try:
                AgentVersion.parse(agent.version)
            except ValueError as exc:
                raise AgentLoadError(str(exc)) from exc
        return agent
