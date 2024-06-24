from typing import Dict, Type

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparametrizable import (
    Hyperparametrizable,
)
from ray.rllib.algorithms import (
    DQN,
    PPO,
    SAC,
    Algorithm,
    AlgorithmConfig,
    DQNConfig,
    PPOConfig,
    SACConfig,
)


class AlgorithmConfigKwargs(Hyperparametrizable):
    hyperparameters = [
        FloatHyperparameter(name="lr", low=1e-5, high=1e-1, log=True, default=0.001),
        FloatHyperparameter(
            name="gamma",
            low=0.0,
            high=1.0,
            default=0.99,
            suggest_low=True,
            suggest_high=True,
        ),
    ]

    @classmethod
    def update_algo_config(cls, config: AlgorithmConfig, **kwargs) -> AlgorithmConfig:
        kwargs = cls.complete_with_default_hyperparameters(kwargs)
        config = config.copy(copy_frozen=False)
        config.training(gamma=kwargs["gamma"], lr=kwargs["lr"])
        return config


class PPOConfigKwargs(AlgorithmConfigKwargs):

    hyperparameters = AlgorithmConfigKwargs.copy_and_update_hyperparameters(
        lr=dict(default=5e-5)
    ) + [
        CategoricalHyperparameter(
            name="shuffle_sequences", choices=[True, False], default=True
        ),
        CategoricalHyperparameter(
            name="use_kl_loss", choices=[True, False], default=True
        ),
        FloatHyperparameter(
            name="kl_coeff",
            low=0.0,
            high=0.5,
            default=0.2,
            depends_on=("use_kl_loss", [True]),
        ),
        FloatHyperparameter(
            name="kl_target",
            low=1e-4,
            high=0.5,
            log=True,
            default=0.01,
            depends_on=("use_kl_loss", [True]),
        ),
    ]

    @classmethod
    def update_algo_config(cls, config: PPOConfig, **kwargs) -> PPOConfig:
        kwargs = cls.complete_with_default_hyperparameters(kwargs)
        config: PPOConfig = super().update_algo_config(config, **kwargs)
        config.training(
            shuffle_sequences=kwargs["shuffle_sequences"],
            use_kl_loss=kwargs["use_kl_loss"],
            kl_coeff=kwargs["kl_coeff"],
            kl_target=kwargs["kl_target"],
        )
        return config


class DQNConfigKwargs(AlgorithmConfigKwargs):
    hyperparameters = AlgorithmConfigKwargs.copy_and_update_hyperparameters(
        lr=dict(default=5e-4)
    ) + [
        CategoricalHyperparameter(name="double_q", choices=[True, False], default=True),
        FloatHyperparameter(
            name="tau",
            low=0.0,
            high=1.0,
            default=1.0,
            suggest_high=True,
            suggest_low=True,
        ),
    ]

    @classmethod
    def update_algo_config(cls, config: DQNConfig, **kwargs) -> DQNConfig:
        kwargs = cls.complete_with_default_hyperparameters(kwargs)
        config: DQNConfig = super().update_algo_config(config, **kwargs)
        config.training(
            double_q=kwargs["double_q"],
            tau=kwargs["tau"],
        )
        return config


class SACConfigKwargs(AlgorithmConfigKwargs):
    hyperparameters = AlgorithmConfigKwargs.copy_and_update_hyperparameters(
        lr=dict(default=5e-4)
    ) + [
        CategoricalHyperparameter(name="twin_q", choices=[True, False], default=True),
        FloatHyperparameter(name="tau", low=1e-4, high=1.0, default=5e-3, log=True),
    ]

    @classmethod
    def update_algo_config(cls, config: SACConfig, **kwargs) -> SACConfig:
        kwargs = cls.complete_with_default_hyperparameters(kwargs)
        config: SACConfig = super().update_algo_config(config, **kwargs)
        config.training(
            twin_q=kwargs["twin_q"],
            tau=kwargs["tau"],
        )
        return config


algo_config_mapping: Dict[Type[Algorithm], Type[AlgorithmConfigKwargs]] = {
    PPO: PPOConfigKwargs,
    DQN: DQNConfigKwargs,
    SAC: SACConfigKwargs,
}
