from tensordict.nn import TensorDictSequential, TensorDictModule
from torchrl.envs import GymEnv

from gp.variationalgp import ExactGaussianProcessRegressor
from util.fetchdevice import fetch_device


def make_gp_q_agent(
                dummy_env: GymEnv,
                gp_model_str: str = "exact_gp_value",
                annealing_num_steps=2000) -> tuple[TensorDictSequential, TensorDictSequential]:
    # Add assertion that dummy environment needs to have discrete action space.

    gp_regressor = ExactGaussianProcessRegressor().to(device=fetch_device())   # Hard code for now.

    value_model = TensorDictModule(
        gp_regressor,
        in_keys=["observation", "action"],    # Concatenate observation and action as input
        out_keys=["action_value"]
    )

    """
    # noinspection PyTypeChecker
    actor = TensorDictSequential(
        value_model,
            action_space=dummy_env.action_spec
        )
    )

    exploration_module = EGreedyModule(
        spec=dummy_env.spec,
        annealing_num_steps=annealing_num_steps,
    )

    # noinspection PyTypeChecker
    actor_explore = TensorDictSequential(
        actor,
        exploration_module
    )

    return actor, actor_explore
    """
