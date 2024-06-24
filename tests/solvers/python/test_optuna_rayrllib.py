from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Type

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    FloatHyperparameter,
    Hyperparameter,
)
from stable_baselines3 import A2C, PPO

from skdecide import Solver
from skdecide.core import Value
from skdecide.hub.domain.maze.maze import Maze, State
from skdecide.hub.solver.astar import Astar
from skdecide.hub.solver.cgp import CGP
from skdecide.hub.solver.iw import IW
from skdecide.hub.solver.mcts import UCT
from skdecide.hub.solver.ray_rllib import RayRLlib
from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.optuna_utils import generic_optuna_experiment_monoproblem
from skdecide.utils import match_solvers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


def test_generic_optuna_experiment_monoproblem_with_ray_rllib():
    # Params of the study
    rollout_num_episodes = 0
    rollout_max_steps_by_episode = 500
    domain_reset_is_deterministic = True
    seed = 42  # set this to an integer to get reproducible results, else to None
    n_trials = 5  # number of trials to launch
    create_another_study = False
    overwrite_study = True
    study_basename = (
        f"maze-rayrllib-{rollout_num_episodes}-{rollout_max_steps_by_episode}"
    )

    # Domain to test
    domain_factory = Maze

    generated_configs = []

    # Wrapper around RayRLlib
    class FakeRayRLlib(RayRLlib):
        def _solve(self) -> None:
            if not hasattr(self, "_algo"):
                self._init_algo()
            generated_configs.append(self._algo.config.to_dict())

    # Solvers to test
    # solver_classes = match_solvers(domain_factory())
    solver_classes = [FakeRayRLlib]
    solver_classes = match_solvers(domain=domain_factory(), candidates=solver_classes)

    # heuristics and state_features (needed by some solvers

    # Fixed kwargs per solver: either hyperparameters we do not want to search, or other parameters like time limits
    kwargs_fixed_by_solver: Dict[Type[Solver], Dict[str, Any]] = defaultdict(
        dict,  # default kwargs for unspecified solvers
        {
            RayRLlib: {"train_iterations": 1},
        },
    )

    # Add new hyperparameters to some solvers
    additional_hyperparameters_by_solver: Dict[
        Type[Solver], List[Hyperparameter]
    ] = defaultdict(
        list,  # default additional hyperparameters for all solvers (empty list)
        {},
    )

    # Restrict some hyperparameters choices, for some solvers (making use of `kwargs_by_name` of `suggest_with_optuna`)
    suggest_optuna_kwargs_by_name_by_solver: Dict[
        Type[Solver], Dict[str, Dict[str, Any]]
    ] = defaultdict(
        dict,  # default kwargs_by_name for unspecified solvers
        {},
    )

    study = generic_optuna_experiment_monoproblem(
        domain_factory=domain_factory,
        solver_classes=solver_classes,
        kwargs_fixed_by_solver=kwargs_fixed_by_solver,
        suggest_optuna_kwargs_by_name_by_solver=suggest_optuna_kwargs_by_name_by_solver,
        additional_hyperparameters_by_solver=additional_hyperparameters_by_solver,
        n_trials=n_trials,
        rollout_num_episodes=rollout_num_episodes,
        rollout_max_steps_by_episode=rollout_max_steps_by_episode,
        domain_reset_is_deterministic=domain_reset_is_deterministic,
        study_basename=study_basename,
        create_another_study=create_another_study,
        overwrite_study=overwrite_study,
        seed=seed,
    )

    for trial, config in zip(study.trials, generated_configs):
        algo_class_name = trial.params["FakeRayRLlib.algo_class"]
        lr = trial.params[f"FakeRayRLlib.config_kwargs_{algo_class_name.lower()}.lr"]
        gamma = trial.params[
            f"FakeRayRLlib.config_kwargs_{algo_class_name.lower()}.gamma"
        ]
        assert config["lr"] == lr
        assert config["gamma"] == gamma
        if algo_class_name in ["DQN", "SAC"]:
            tau = trial.params[
                f"FakeRayRLlib.config_kwargs_{algo_class_name.lower()}.tau"
            ]
            assert config["tau"] == tau
        else:
            assert (
                f"FakeRayRLlib.config_kwargs_{algo_class_name.lower()}.tau"
                not in trial.params
            )
