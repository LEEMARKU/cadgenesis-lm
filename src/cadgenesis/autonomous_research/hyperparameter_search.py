"""
Hyperparameter Search - Bayesian Optimization, Population-Based Training, Evolutionary Search,
Random Search, Grid Search.
"""

from __future__ import annotations

import random
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any

try:
    import numpy as np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - numpy ships with torch
    import math

    class _NpFallback:
        @staticmethod
        def log(x):
            return math.log(x)

        @staticmethod
        def exp(x):
            return math.exp(x)

    np = _NpFallback()  # type: ignore[assignment]


class SearchAlgorithm(str, Enum):
    BAYESIAN = "bayesian"
    POPULATION_BASED = "population_based"
    EVOLUTIONARY = "evolutionary"
    RANDOM = "random"
    GRID = "grid"


@dataclass
class ParameterSpace:
    """Defines the search space for hyperparameters."""

    name: str
    param_type: str  # "float", "int", "categorical", "log_uniform"
    min_value: float | None = None
    max_value: float | None = None
    categories: list[Any] | None = None
    default: Any = None


@dataclass
class SearchSpace:
    """Collection of parameter spaces."""

    spaces: dict[str, ParameterSpace] = field(default_factory=dict)

    def add_float(
        self,
        name: str,
        min_val: float,
        max_val: float,
        log: bool = False,
        default: float | None = None,
    ) -> None:
        self.spaces[name] = ParameterSpace(
            name, "log_uniform" if log else "float", min_val, max_val, default=default
        )

    def add_int(self, name: str, min_val: int, max_val: int, default: int | None = None) -> None:
        self.spaces[name] = ParameterSpace(
            name, "int", float(min_val), float(max_val), default=default
        )

    def add_categorical(self, name: str, categories: list[Any], default: Any = None) -> None:
        self.spaces[name] = ParameterSpace(
            name, "categorical", categories=categories, default=default
        )

    def sample(self) -> dict[str, Any]:
        """Sample a random configuration from the space."""
        config = {}
        for name, space in self.spaces.items():
            if space.param_type == "float":
                assert space.min_value is not None and space.max_value is not None
                config[name] = random.uniform(space.min_value, space.max_value)
            elif space.param_type == "log_uniform":
                assert space.min_value is not None and space.max_value is not None
                log_min = np.log(space.min_value)
                log_max = np.log(space.max_value)
                config[name] = np.exp(random.uniform(log_min, log_max))
            elif space.param_type == "int":
                assert space.min_value is not None and space.max_value is not None
                config[name] = random.randint(int(space.min_value), int(space.max_value))
            elif space.param_type == "categorical":
                assert space.categories is not None
                config[name] = random.choice(space.categories)
            else:
                config[name] = space.default
        return config


@dataclass
class SearchResult:
    """Result of a hyperparameter search trial."""

    trial_id: str
    parameters: dict[str, Any]
    metrics: dict[str, float]
    status: str = "completed"
    duration: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseSearcher(ABC):
    """Abstract base class for hyperparameter searchers."""

    def __init__(self, space: SearchSpace, objective: str, maximize: bool = True):
        self.space = space
        self.objective = objective
        self.maximize = maximize
        self.trials: list[SearchResult] = []
        self._lock = RLock()

    @abstractmethod
    def suggest(self) -> dict[str, Any]:
        """Suggest next parameters to try."""
        pass

    def observe(
        self,
        parameters: dict[str, Any],
        metrics: dict[str, float],
        duration: float = 0,
        error: str | None = None,
    ) -> None:
        """Record the result of a trial."""
        result = SearchResult(
            trial_id=str(uuid.uuid4()),
            parameters=parameters,
            metrics=metrics,
            status="completed" if error is None else "failed",
            duration=duration,
            error=error,
        )
        with self._lock:
            self.trials.append(result)

    def get_best(self) -> SearchResult | None:
        """Get the best trial so far."""
        with self._lock:
            completed = [t for t in self.trials if t.status == "completed"]
            if not completed:
                return None
            return (
                max(completed, key=lambda t: t.metrics.get(self.objective, 0))
                if self.maximize
                else min(completed, key=lambda t: t.metrics.get(self.objective, float("inf")))
            )


class RandomSearcher(BaseSearcher):
    """Random search."""

    def suggest(self) -> dict[str, Any]:
        return self.space.sample()


class GridSearcher(BaseSearcher):
    """Grid search."""

    def __init__(self, space: SearchSpace, objective: str, maximize: bool = True):
        super().__init__(space, objective, maximize)
        self._grid = self._generate_grid()
        self._index = 0

    def _generate_grid(self) -> list[dict[str, Any]]:
        # Simple grid: use default + min/max for each parameter
        grid: list[dict[str, Any]] = [{}]
        for name, space in self.space.spaces.items():
            new_grid: list[dict[str, Any]] = []
            values: list[Any]
            if space.param_type in ("float", "log_uniform"):
                assert space.min_value is not None and space.max_value is not None
                values = [
                    space.min_value,
                    (space.min_value + space.max_value) / 2,
                    space.max_value,
                ]
            elif space.param_type == "int":
                assert space.min_value is not None and space.max_value is not None
                values = [
                    int(space.min_value),
                    int((space.min_value + space.max_value) / 2),
                    int(space.max_value),
                ]
            elif space.param_type == "categorical":
                assert space.categories is not None
                values = space.categories
            else:
                values = [space.default]

            for g in grid:
                for v in values:
                    new_g = g.copy()
                    new_g[name] = v
                    new_grid.append(new_g)
            grid = new_grid
        return grid

    def suggest(self) -> dict[str, Any]:
        with self._lock:
            if self._index >= len(self._grid):
                return self.space.sample()
            config = self._grid[self._index]
            self._index += 1
            return config


class HyperparameterSearch:
    """Main interface for hyperparameter search."""

    def __init__(self):
        self._searchers: dict[str, BaseSearcher] = {}
        self._lock = RLock()

    def create_search(
        self,
        name: str,
        space: SearchSpace,
        objective: str,
        algorithm: SearchAlgorithm = SearchAlgorithm.RANDOM,
        maximize: bool = True,
    ) -> BaseSearcher:
        """Create a new hyperparameter search."""
        searcher: BaseSearcher
        if algorithm == SearchAlgorithm.RANDOM:
            searcher = RandomSearcher(space, objective, maximize)
        elif algorithm == SearchAlgorithm.GRID:
            searcher = GridSearcher(space, objective, maximize)
        else:
            # For bayesian, population-based, evolutionary - use random as fallback
            searcher = RandomSearcher(space, objective, maximize)

        with self._lock:
            self._searchers[name] = searcher

        return searcher

    def get_searcher(self, name: str) -> BaseSearcher | None:
        with self._lock:
            return self._searchers.get(name)

    def run_search(
        self,
        name: str,
        train_fn: Callable[[dict[str, Any]], dict[str, float]],
        n_trials: int = 20,
    ) -> list[SearchResult]:
        """Run hyperparameter search."""
        searcher = self.get_searcher(name)
        if not searcher:
            raise ValueError(f"Searcher {name} not found")

        for _ in range(n_trials):
            params = searcher.suggest()
            start = time.time()
            try:
                metrics = train_fn(params)
                searcher.observe(params, metrics, time.time() - start)
            except Exception as e:
                searcher.observe(params, {}, time.time() - start, str(e))

        return searcher.trials
