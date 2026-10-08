import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import ray
import torch as th
from pytest_cases import fixture
from ray.rllib.algorithms import PPO

from skdecide.hub.solver.ray_rllib import RayRLlib
from skdecide.hub.solver.ray_rllib.autoregressive.algorithms.ppo.torch.autoregressive_ppo_torch_rl_module import (
    AutoregressivePPOTorchRLModule,
)
from skdecide.hub.solver.ray_rllib.ray_rllib import SK_DEFAULT_MODULE_ID
from skdecide.hub.solver.utils.gnn.torch_layers import Graph2NodeLayer
from skdecide.hub.solver.utils.torch.utils import extract_module_parameters_values
from skdecide.utils import rollout


@fixture
def ray_init():
    # add module test_gnn_ray_rllib and thus GraphMaze to ray runtimeenv
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"working_dir": os.path.abspath(f"{os.path.dirname(__file__)}/..")},
    )


@fixture
def ppo_config():
    return (
        PPO.get_default_config()
        .env_runners(
            # set num of CPU<1 to avoid hanging for ever in github actions on macos 11)
            num_cpus_per_env_runner=0.5
        )
        # small number to increase speed of the unit test
        .training(minibatch_size=16, train_batch_size_per_learner=32)
        # uncomment next lines to run in local mode and debug more easily
        .env_runners(num_env_runners=0)
        .learners(num_learners=0)
    )


def test_autoregressive_ppo(graph_walk_domain_factory, ppo_config):
    domain_factory = graph_walk_domain_factory
    with RayRLlib(
        domain_factory=domain_factory,
        algo_class=PPO,
        config=ppo_config,
        train_iterations=1,
        autoregressive_action=True,
    ) as solver:
        # store initial weigths to see if they update during training
        solver._init_algo()
        rl_module = solver.get_policy()[SK_DEFAULT_MODULE_ID]
        assert isinstance(rl_module, AutoregressivePPOTorchRLModule)
        value_init_params = extract_module_parameters_values(rl_module.vf)
        action_init_params = [
            extract_module_parameters_values(pi_component)
            for pi_component in rl_module.pi_per_component
        ]
        solver.solve()
        # rl_module = solver.get_policy()[SK_DEFAULT_MODULE_ID]
        # assert isinstance(rl_module, AutoregressivePPOTorchRLModule)
        value_final_params = extract_module_parameters_values(rl_module.vf)
        action_final_params = [
            extract_module_parameters_values(pi_component)
            for pi_component in rl_module.pi_per_component
        ]
        max_steps = 20
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=max_steps,
            num_episodes=1,
            render=False,
        )

        # check policy params updated
        for i in range(len(action_init_params)):
            assert not (
                all(
                    np.allclose(
                        action_init_params[i][name], action_final_params[i][name]
                    )
                    for name in action_init_params[i]
                )
            ), f"action net #{i} params not updated"
        # assert not (
        #     all(
        #         np.allclose(value_init_params[name], value_final_params[name])
        #         for name in value_init_params
        #     )
        # ), f"value net params not updated"


@pytest.mark.skip("Not yet ready")
def test_autoregressive_graph_ppo(graph_walk_with_graph_obs_domain_factory, ppo_config):
    domain_factory = graph_walk_with_graph_obs_domain_factory
    with RayRLlib(
        domain_factory=domain_factory,
        algo_class=PPO,
        config=ppo_config,
        train_iterations=1,
        autoregressive_action=True,
    ) as solver:
        # pre-init algo (done normally during solve()) to extract init weights
        solver._init_algo()
        value_init_params = extract_module_parameters_values(
            solver._algo.policy.value_net
        )
        gnn_features_init_params = extract_module_parameters_values(
            solver._algo.policy.features_extractor
        )
        action_init_params = [
            extract_module_parameters_values(action_net)
            for action_net in solver._algo.policy.action_nets
        ]
        # solve
        solver.solve()
        # compare final parameters to see if actually updated during training
        value_final_params = extract_module_parameters_values(
            solver._algo.policy.value_net
        )
        gnn_features_final_params = extract_module_parameters_values(
            solver._algo.policy.features_extractor
        )
        action_final_params = [
            extract_module_parameters_values(action_net)
            for action_net in solver._algo.policy.action_nets
        ]

        assert not (
            all(
                np.allclose(value_init_params[name], value_final_params[name])
                for name in value_init_params
            )
        ), f"value net params not updated"
        assert not (
            all(
                np.allclose(
                    gnn_features_init_params[name], gnn_features_final_params[name]
                )
                for name in gnn_features_final_params
            )
        ), f"value net params not updated"
        for i in range(len(action_init_params)):
            assert not (
                all(
                    np.allclose(
                        action_init_params[i][name], action_final_params[i][name]
                    )
                    for name in action_init_params[i]
                )
            ), f"action net #{i} params not updated"

        # rollout
        max_steps = 20
        episodes = rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=max_steps,
            num_episodes=1,
            render=False,
            return_episodes=True,
        )
        observations, actions, values = episodes[0]
        assert (
            len(actions) < max_steps - 1
        )  # optimal would be 2, but not always found...


@pytest.mark.skip("Not yet ready")
def test_autoregressive_graph2node_ppo(
    graph_walk_with_graph_obs_domain_factory, ppo_config
):
    domain_factory = graph_walk_with_graph_obs_domain_factory
    with RayRLlib(
        domain_factory=domain_factory,
        algo_class=PPO,
        config=ppo_config,
        train_iterations=1,
        autoregressive_action=True,
        graph_node_action=True,
    ) as solver:
        # pre-init algo (done normally during solve()) to extract init weights
        solver._init_algo()
        for i in range(len(domain_factory()._gym_env.action_space.nvec)):
            if i == 0:
                assert isinstance(solver._algo.policy.action_nets[i], th.nn.Linear)
            else:
                assert isinstance(solver._algo.policy.action_nets[i], Graph2NodeLayer)
        value_init_params = extract_module_parameters_values(
            solver._algo.policy.value_net
        )
        gnn_features_init_params = extract_module_parameters_values(
            solver._algo.policy.features_extractor
        )
        action_init_params = [
            extract_module_parameters_values(action_net)
            for action_net in solver._algo.policy.action_nets
        ]

        # solve
        solver.solve()

        # compare final parameters to see if actually updated during training
        value_final_params = extract_module_parameters_values(
            solver._algo.policy.value_net
        )
        gnn_features_final_params = extract_module_parameters_values(
            solver._algo.policy.features_extractor
        )
        action_final_params = [
            extract_module_parameters_values(action_net)
            for action_net in solver._algo.policy.action_nets
        ]

        assert not (
            all(
                np.allclose(value_init_params[name], value_final_params[name])
                for name in value_init_params
            )
        ), f"value net params not updated"
        assert not (
            all(
                np.allclose(
                    gnn_features_init_params[name], gnn_features_final_params[name]
                )
                for name in gnn_features_final_params
            )
        ), f"value net params not updated"
        for i in range(len(action_init_params)):
            assert not (
                all(
                    np.allclose(
                        action_init_params[i][name], action_final_params[i][name]
                    )
                    for name in action_init_params[i]
                )
            ), f"action net #{i} params not updated"

        # rollout
        max_steps = 20
        episodes = rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=max_steps,
            num_episodes=1,
            render=False,
            return_episodes=True,
        )
        observations, actions, values = episodes[0]
        assert (
            len(actions) < max_steps - 1
        )  # optimal would be 2, but not always found...


@pytest.mark.skip("Not yet ready")
def test_autoregressive_heterograph2node_ppo(
    graph_walk_with_heterograph_obs_domain_factory,
    action_components_node_flag_indices,
    ppo_config,
):
    domain_factory = graph_walk_with_heterograph_obs_domain_factory
    model_id = "model_id"
    agent2module_id = {agent: model_id for agent in domain_factory().get_agents()}
    with RayRLlib(
        domain_factory=domain_factory,
        algo_class=PPO,
        config=ppo_config,
        train_iterations=1,
        autoregressive_action=True,
        model_configs={
            model_id: dict(
                action_components_node_flag_indices=action_components_node_flag_indices,
            )
        },
        agent2module_id=agent2module_id,
    ) as solver:
        # pre-init algo (done normally during solve()) to extract init weights
        solver._init_algo()
        for i in range(len(domain_factory()._gym_env.action_space.nvec)):
            if i == 0:
                assert isinstance(solver._algo.policy.action_nets[i], th.nn.Linear)
            else:
                assert isinstance(solver._algo.policy.action_nets[i], Graph2NodeLayer)
        value_init_params = extract_module_parameters_values(
            solver._algo.policy.value_net
        )
        gnn_features_init_params = extract_module_parameters_values(
            solver._algo.policy.features_extractor
        )
        action_init_params = [
            extract_module_parameters_values(action_net)
            for action_net in solver._algo.policy.action_nets
        ]

        # solve
        solver.solve()

        # compare final parameters to see if actually updated during training
        value_final_params = extract_module_parameters_values(
            solver._algo.policy.value_net
        )
        gnn_features_final_params = extract_module_parameters_values(
            solver._algo.policy.features_extractor
        )
        action_final_params = [
            extract_module_parameters_values(action_net)
            for action_net in solver._algo.policy.action_nets
        ]

        assert not (
            all(
                np.allclose(value_init_params[name], value_final_params[name])
                for name in value_init_params
            )
        ), f"value net params not updated"
        assert not (
            all(
                np.allclose(
                    gnn_features_init_params[name], gnn_features_final_params[name]
                )
                for name in gnn_features_final_params
            )
        ), f"value net params not updated"
        for i in range(len(action_init_params)):
            assert not (
                all(
                    np.allclose(
                        action_init_params[i][name], action_final_params[i][name]
                    )
                    for name in action_init_params[i]
                )
            ), f"action net #{i} params not updated"

        # rollout
        max_steps = 20
        episodes = rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=max_steps,
            num_episodes=1,
            render=False,
            return_episodes=True,
        )
        observations, actions, values = episodes[0]
        assert (
            len(actions) < max_steps - 1
        )  # optimal would be 2, but not always found...


@pytest.mark.skip("Not yet ready")
def test_autoregressive_heterograph2node_ppo_save_load(
    graph_walk_with_heterograph_obs_domain_factory,
    graph_walk_with_heterograph_obs_domain_factory_2,
    action_components_node_flag_indices,
    ppo_config,
):
    domain_factory = graph_walk_with_heterograph_obs_domain_factory
    domain_factory_2 = graph_walk_with_heterograph_obs_domain_factory_2

    model_id = "model_id"
    with TemporaryDirectory() as directory:
        path = f"{directory}/sb3"
        agent2module_id = {agent: model_id for agent in domain_factory().get_agents()}
        with RayRLlib(
            domain_factory=domain_factory,
            algo_class=PPO,
            config=ppo_config,
            train_iterations=1,
            autoregressive_action=True,
            model_configs={
                model_id: dict(
                    action_components_node_flag_indices=action_components_node_flag_indices,
                )
            },
            agent2module_id=agent2module_id,
        ) as solver:
            # solve
            solver.solve()
            # store
            solver.save(path)

        # load for another domain (more objects) + solve + rollout
        agent2module_id = {agent: model_id for agent in domain_factory_2().get_agents()}
        with RayRLlib(
            domain_factory=domain_factory_2,
            algo_class=PPO,
            config=ppo_config,
            train_iterations=1,
            autoregressive_action=True,
            model_configs={
                model_id: dict(
                    action_components_node_flag_indices=action_components_node_flag_indices,
                    n_graph2node_components=1,
                )
            },
            agent2module_id=agent2module_id,
        ) as solver:
            # load
            solver.load(path)
            assert solver._algo.n_steps == 15

            # solve
            solver.solve()

            # rollout on domain 2
            max_steps = 20
            episodes = rollout(
                domain=graph_walk_with_heterograph_obs_domain_factory_2(),
                solver=solver,
                max_steps=max_steps,
                num_episodes=1,
                render=False,
                return_episodes=True,
            )
