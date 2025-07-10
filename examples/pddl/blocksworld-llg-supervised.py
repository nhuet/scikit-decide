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

from skdecide import EnvironmentOutcome, rollout
from skdecide.hub.domain.plado import ActionEncoding, PladoPddlDomain, StateEncoding
from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.hub.solver.stable_baselines.autoregressive.ppo.autoregressive_ppo import (
    AutoregressiveGraphPPO,
)

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
for supervised in [True, False]:
    if supervised:
        print("# SUPERVISED ")
    else:
        print("# UNSUPERVISED ")
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=AutoregressiveGraphPPO,
        baselines_policy="HeteroGraph2NodePolicy",
        policy_kwargs=dict(
            action_components_node_flag_indices=action_components_node_flag_indices
        ),
        autoregressive_action=True,
        learn_config={"total_timesteps": 1000},
        n_steps=200,
        supervised=supervised,
        plan=optimal_plan,
    ) as solver:
        solver.solve()
        max_steps = 50
        episodes = rollout(
            domain=domain,
            solver=solver,
            max_steps=max_steps,
            num_episodes=10,
            render=False,
            return_episodes=True,
            outcome_formatter=lambda outcome: outcome_formater(outcome, domain),
            observation_formatter=lambda obs: observation_formater(obs, domain),
            verbose=False,
        )
        print([len(episode[1]) for episode in episodes])
