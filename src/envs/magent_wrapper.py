from pathlib import Path
import importlib

import gymnasium as gym
from gymnasium.spaces import Tuple

import magent2


class MAgent2Wrapper(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
    }
    def __init__(self, env_name, map_size=30, **kwargs):
        env_kwargs = {"map_size": map_size}
        env_kwargs.update(kwargs)
        env_module = importlib.import_module(f"magent2.environments.{env_name}")
        self._env = env_module.parallel_env(**env_kwargs)
        self._env.reset()

        self.n_agents = self._env.num_agents
        self.last_obs = None

        self.action_space = Tuple(
            tuple([self._env.action_space(k) for k in self._env.agents])
        )
        self.observation_space = Tuple(
            tuple([self._env.observation_space(k) for k in self._env.agents])
        )

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        obs = tuple([obs[k] for k in self._env.agents])
        self.last_obs = obs
        return obs, info

    def render(self):
        return self._env.render()

    def step(self, actions):
        dict_actions = {}
        for agent, action in zip(self._env.agents, actions):
            dict_actions[agent] = action

        observations, rewards, dones, truncated, infos = self._env.step(dict_actions)

        obs = tuple([observations[k] for k in self._env.agents])
        rewards = [rewards[k] for k in self._env.agents]
        done = all([dones[k] for k in self._env.agents])
        truncated = all([truncated[k] for k in self._env.agents])
        info = {
            f"{k}_{key}": value
            for k in self._env.agents
            for key, value in infos[k].items()
        }
        if done:
            assert len(obs) == 0
            assert len(rewards) == 0
            obs = self.last_obs
            rewards = [0] * len(obs)
        else:
            self.last_obs = obs
        return obs, rewards, done, truncated, info

    def close(self):
        return self._env.close()

envs = Path(magent2.__path__[0]).glob("**/*_v?.py")
for e in envs:
    name = e.stem.replace("_", "-")
    filename = e.stem

    gymkey = f"magent2-{name}"
    gym.register(
        gymkey,
        entry_point="envs.magent_wrapper:MAgent2Wrapper",
        kwargs={
            "env_name": filename,
            "map_size": 30
        },
    )
