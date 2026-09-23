"""Replace only XPolicy's transport when EMERGE drives the official EvalEnv."""
from contextlib import contextmanager


class DirectAgentTransport:
    """Pi0.5 requests go through the driver's OpenPI client, not XPolicy RPC."""
    def __init__(self, **_kwargs):
        pass

    def call(self, func_name, **_kwargs):
        if func_name != "reset":
            raise RuntimeError(f"Unexpected XPolicy RPC in direct agent mode: {func_name}")
        # The OpenPI policy is stateless across requests. Episode workspaces
        # and agent sessions are independently reset by the worker.
        return None


@contextmanager
def direct_agent_transport(module):
    previous = module.WsModelClient
    module.WsModelClient = DirectAgentTransport
    try:
        yield
    finally:
        module.WsModelClient = previous


def create_agent_eval_env(config, app):
    from src.eval_client import eval_env
    with direct_agent_transport(eval_env):
        return eval_env.create_eval_env(config, app)
