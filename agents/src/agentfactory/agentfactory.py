from agent.agent import Agent


class AgentFactory:
    """
    Naive factory method implementation for
    RL agent creation.
    """

    @staticmethod
    def create_agent(agent_type: str) -> Agent:
        """
        Factory method for Agent creation.
        :param agent_type: a string key corresponding to the agent.
        :return: an object of type Agent.
        """
        raise ValueError("Invalid model type. Supported types: 'linear', 'mlp'.")
