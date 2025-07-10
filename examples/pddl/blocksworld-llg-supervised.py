"""Blocksworld with 3 objects with graph-objects representation + sb3 autoregressive solver.

Blocksworld: https://en.wikipedia.org/wiki/Blocks_world
Lifted Learning Graph representation: see
    Chen, D. Z., Thiébaux, S., & Trevizan, F. (2024).
    Learning Domain-Independent Heuristics for Grounded and Lifted Planning.
    Proceedings of the AAAI Conference on Artificial Intelligence, 38(18), 20078-20086.
    https://doi.org/10.1609/aaai.v38i18.29986
sb3-autoregressive: components of the action are predicted one by one,
    taking into account the choices of the previous components.
    components that are nodes of the observation graph are deduced by a GNN
    (and value is also predicted thanks to a GNN used for feature extraction)

"""


import os

import numpy as np
import torch as th
import torch.nn.functional as F
import torch_geometric as thg

from skdecide import EnvironmentOutcome, rollout
from skdecide.hub.domain.plado import ActionEncoding, PladoPddlDomain, StateEncoding
from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.hub.solver.stable_baselines.autoregressive.ppo.autoregressive_ppo import (
    AutoregressiveGraphPPO,
)
from skdecide.hub.solver.utils.gnn.advanced_gnn import AdvancedGNN

pddl_examples_dir = os.path.dirname(os.path.abspath(__file__))
pddl_domains_def_dir = os.path.abspath(
    f"{pddl_examples_dir}/../../tests/domains/python/pddl_domains"
)
domain_problem_dirpath = f"{pddl_domains_def_dir}/blocks"
domain_path = f"{domain_problem_dirpath}/domain.pddl"
problem_path = f"{domain_problem_dirpath}/probBLOCKS-3-0.pddl"


def outcome_formater(outcome: EnvironmentOutcome, domain: PladoPddlDomain) -> str:
    return f"observation={domain.repr_obs_as_plado(outcome.observation)}, value={outcome.value}, termination={outcome.termination}, info={outcome.info}"


def observation_formater(
    observation: PladoPddlDomain.T_observation, domain: PladoPddlDomain
) -> str:
    return domain.repr_obs_as_plado(observation)


domain_factory = lambda: PladoPddlDomain(
    domain_path=domain_path,
    problem_path=problem_path,
    state_encoding=StateEncoding.GYM_GRAPH_LLG,
    action_encoding=ActionEncoding.GYM_MULTIDISCRETE,
)

domain = domain_factory()

# learn "supervised"

optimal_plan = [
    (3, 0, 1),  # unstack a b
    (1, 0, -1),  # put-down a
    (0, 1, -1),  # put-up b
    (2, 1, 2),  # stack b c
]

# Action components via GNN -> node (actions or objects) of the llg graph
action_components_node_flag_indices = domain.get_action_components_node_flag_indices()
gnn_hidden_channels = 16
gnn_n_layers = 7
solver = StableBaseline(
    domain_factory=domain_factory,
    algo_class=AutoregressiveGraphPPO,
    baselines_policy="HeteroGraph2NodePolicy",
    policy_kwargs=dict(
        action_components_node_flag_indices=action_components_node_flag_indices,
        action_gnn_class=AdvancedGNN,  # Graph2NodeLayer's gnn_class
        action_gnn_kwargs=dict(  # Graph2NodeLayer's gnn_kwargs
            # in_channels automatically filled by Graph2NodeLayer
            # out_channels automatically filled by Graph2NodeLayer
            hidden_channels=gnn_hidden_channels,
            num_layers=gnn_n_layers,
            # dropout=0.2,
            message_passing_cls=thg.nn.GCNConv,
            supports_edge_weight=True,
            supports_edge_attr=False,
            using_encoder=True,
            using_decoder=True,
        ),
        features_extractor_kwargs=dict(
            # kwargs for GraphFeaturesExtractor
            gnn_class=AdvancedGNN,
            gnn_kwargs=dict(
                # in_channels automatically filled by GraphFeaturesExtractor
                hidden_channels=gnn_hidden_channels,  # output_dim as out_channels=None
                num_layers=gnn_n_layers,
                # dropout=0.2,
                message_passing_cls=thg.nn.GCNConv,
                supports_edge_weight=True,
                supports_edge_attr=False,
                using_encoder=True,
            ),
            gnn_out_dim=gnn_hidden_channels,  # correspond to GNN ouput dim
        ),
    ),
    autoregressive_action=True,
    learn_config={"total_timesteps": 1000},
    n_steps=200,
    supervised=True,
    plan=optimal_plan,
)

# extract the policy (and create the algo on the way)
policy = solver.get_policy()

learning_env = solver._learning_env
algo = solver._algo
learning_vecenv = algo._wrap_env(learning_env)


def prepare_evaluate_actions_inputs(obs, algo, learning_vecenv):
    action_masks = algo.get_action_masks(learning_vecenv)
    expected_actions = algo.get_expected_actions(learning_vecenv)
    tmp_rollout_buffer = algo.rollout_buffer_class(
        buffer_size=1,
        observation_space=algo.observation_space,
        action_space=algo.action_space,
        device=algo.device,
        gamma=algo.gamma,
        gae_lambda=algo.gae_lambda,
        n_envs=1,
        **algo.rollout_buffer_kwargs,
    )
    tmp_obs = obs
    tmp_actions = expected_actions
    tmp_rewards = np.zeros((1, 1))
    tmp_episode_starts = np.ones((1, 1), dtype=bool)
    tmp_values = th.zeros((1, 1))
    tmp_log_probs = th.zeros((1, 1))
    tmp_rollout_buffer.add(
        tmp_obs,
        tmp_actions,
        tmp_rewards,
        tmp_episode_starts,
        tmp_values,
        tmp_log_probs,
        action_masks=action_masks,
    )
    sample = next(tmp_rollout_buffer.get(1))
    return (
        sample.observations,
        sample.actions,
        sample.action_masks,
    )


def train_loop(
    inputs, rewards, policy, optimizer, n_dataset=1000, n_batch_log=100, batch_size=1
):
    policy.train()
    sampler = th.utils.data.RandomSampler(
        inputs, replacement=True, num_samples=n_dataset
    )
    for batch, i_sample in enumerate(sampler):
        obs_tensor, action_tensor, action_mask_tensor = inputs[i_sample]
        reward = rewards[i_sample]

        # Compute logp
        value, log_prob, entropy = policy.evaluate_actions(
            obs_tensor, action_tensor, action_masks=action_mask_tensor
        )
        loss = -log_prob + 1e-2 * F.mse_loss(value, reward)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % n_batch_log == 0:
            loss, current = loss.item(), batch * batch_size + len(inputs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{n_dataset:>5d}]")


# get training inputs corresponding to plan
inputs_train = []
rewards_tensor_train = []
obs = learning_vecenv.reset()
done = False
obs_tensor, action_tensor, action_mask_tensor = prepare_evaluate_actions_inputs(
    obs, algo, learning_vecenv
)
inputs_train.append((obs_tensor, action_tensor, action_mask_tensor))
while not done:
    action = algo.get_expected_actions(learning_vecenv)
    obs, rewards, dones, infos = learning_vecenv.step(action)
    rewards_tensor_train.append(th.as_tensor([[rewards]], device=algo.device))
    done = all(dones)
    if not done:
        inputs_train.append(prepare_evaluate_actions_inputs(obs, algo, learning_vecenv))


# train
optimizer = th.optim.SGD(policy.parameters())
train_loop(inputs_train, rewards_tensor_train, policy, optimizer)


# deterministic rollout
solver.deterministic_prediction = True
episodes = rollout(
    domain=domain_factory(),
    solver=solver,
    max_steps=50,
    num_episodes=1,
    render=False,
    return_episodes=True,
    observation_formatter=None,
    outcome_formatter=None,
)
solver.deterministic_prediction = False

# solve from there
solver.supervised = False  # avoid staying on plan during exploration
solver.solve()
# deterministic rollout
solver.deterministic_prediction = True
episodes = rollout(
    domain=domain_factory(),
    solver=solver,
    max_steps=50,
    num_episodes=1,
    render=False,
    return_episodes=True,
    observation_formatter=None,
    outcome_formatter=None,
)
solver.deterministic_prediction = False
# standard rollout
episodes = rollout(
    domain=domain_factory(),
    solver=solver,
    max_steps=50,
    num_episodes=10,
    render=False,
    return_episodes=True,
    observation_formatter=None,
    outcome_formatter=None,
)
print("length of episodes:", [len(episode[1]) for episode in episodes])
