from gym.envs.registration import register
import sys
sys.modules[__name__]

register(
    id='hedge-v0',
    entry_point='__main__:DeltaHedge',
    timestep_limit=1000,
)
