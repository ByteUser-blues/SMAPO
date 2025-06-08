try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pydantic import Extra

from learning.utils_common import AlgoBase
from planning.replan_algo import RePlanBase, NoPathSoRandomOrStayWrapper, FixNonesWrapper


class RePlanConfig(AlgoBase, extra=Extra.forbid):
    name: Literal['Randomized A*'] = 'Randomized A*'
    num_process: int = 5
    no_path_random: bool = True
    fix_nones: bool = True
    ignore_other_agents: float = 1.0
    cost_penalty_coefficient: float = 0.4
    device: str = 'cpu'


class RePlan:
    def __init__(self, cfg: RePlanConfig = RePlanConfig()):
        self.cfg = cfg
        self.agent = None
        self.fix_nones = cfg.fix_nones
        self.no_path_random = cfg.no_path_random
        self.env = None

    def act(self, observations, rewards=None, dones=None, info=None, skip_agents=None):
        return self.agent.act(observations, skip_agents)

    def after_step(self, dones):
        if all(dones):
            self.agent = None

    def after_reset(self):
        self.reset_states()

    def get_path(self):
        x = self.agent.get_path()
        return x

    def reset_states(self, ):
        self.agent = RePlanBase(seed=self.cfg.seed, ignore_other_agents=self.cfg.ignore_other_agents, cost_penalty_coefficient=self.cfg.cost_penalty_coefficient)
        if self.no_path_random:
            self.agent = NoPathSoRandomOrStayWrapper(self.agent)
        elif self.fix_nones:
            self.agent = FixNonesWrapper(self.agent)

    @staticmethod
    def get_additional_info():
        return {"rl_used": 0.0}
