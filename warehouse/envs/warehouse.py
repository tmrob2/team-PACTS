from optparse import Option
from gym import spaces
import numpy as np
from typing import Optional, List, Tuple
from gym.utils.renderer import Renderer
import gym
import ce
import random

class Warehouse(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "single_rgb_array"], "render_fps": 1}

    def __init__(
        self, 
        initial_agent_loc: List[Tuple],
        feedpoints: List[Tuple],
        nagents: int = 1,
        render_mode: Optional[str] = None, 
        size = 20,
        seed = 4321
        ):
        actions_to_dir = [[1, 0],[0, 1],[-1, 0],[0, -1]]
        # use the initialisation function to setup the Rust version of the warehouse 
        # as well
        self.warehouse_api = ce.Warehouse(
            size, nagents, feedpoints, initial_agent_loc, actions_to_dir, seed
        )
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.xsize = size # the width of a grid
        self.ysize = size # the height of a grid
        self.window_size = 512 # the size of the pygame window

        self.agent_obs = [{"a": aloc, "c": 0, "r": None} for aloc in initial_agent_loc]
        self.agent_rack_positions = [None] * nagents
        self.orig_rack_positions = [None] * nagents
        self.states = [(initial_agent_loc[i], 0, None) for i in range(nagents)]
        # agent locations

        # Observations are dictionaries with the agent's and target's locatoin
        # each location is encoded as an element  {0,..., size}^2 i.e. MultiDiscrete([size,size])

        self.observation_space = spaces.Dict({i:spaces.Dict({
            "a": spaces.Box(np.array([0, 0]), np.array([size - 1, size -1]), shape=(2,), dtype=int),
            "c": spaces.Discrete(2),
        }) for i in range(nagents)}) # r could also be none

        # Actions need to also be instantiated in the OpenAI gym way
        # there are four actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(7)

        """
        The following dictionary maps abstract actions from self.action_space to 
        the directions we will walk in if that action is taken
        I.e. 0 corresponds to right, so on..
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1])
        }

        """
        If human rendering is used, "self.window" will be a reference to the window that we draw to
        'self.clock' will be a clock that is used to ensure that the environment is rendered at the
        correct framerate in human-mode.
        """
        if self.render_mode == "human":
            import pygame

            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        # the following line uses the util class Renderer to gather a collection of frames
        # using a method that computes a single frame. We will define _render_frame below
        self.renderer = Renderer(self.render_mode, self._render_frame)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_locations = self.warehouse_api.agent_initial_locs

        self.renderer.reset()
        self.renderer.render_step()

        observations = {}
        # for each agent return the observation
        for i, agent_pos in enumerate(self.warehouse_api.agent_initial_locs):
            observations[i] = {"a": agent_pos, "c": 0}
            self.agent_rack_positions[i] = None
            self.orig_rack_positions[i] = None

        return observations

    
    def step(self, action: List): # action list will come directly from a policy 
        # the current state is self.state
        observations = {}
        rewards = []
        dones = []
        info = {}

        for agent in range(self.warehouse_api.nagents):
            # get the current agent state
            # The return from the warehouse call will be a List[(state, prob, word)]
            # but for a gym environment we need to pick one of those weights 
            # according to the probability distributed returned.
            print("state: ", self.states[agent])
            v = self.warehouse_api.step(self.states[agent], action[agent])
            print("return from Rust", v)
            # choose one of the v, 
            ind = random.choices(
                list(range(len(v))), weights=[sprime[1] for sprime in v]
            )
            ind0 = ind[0]
            observations[agent] = {"a": v[ind0][0][0], "c": v[ind0][0][1]}
            self.agent_obs[agent] = observations[agent]
            self.agent_rack_positions[agent] = v[ind0][0][2]
            if self.orig_rack_positions[agent] is None:
                self.orig_rack_positions[agent] = self.agent_rack_positions[agent]
            if self.orig_rack_positions[agent] is not None:
                if v[ind0][2][-1] == "D" and v[ind0][0][0] == self.orig_rack_positions[agent]:
                    self.orig_rack_positions[agent] = None
            info["word"] = v[ind0][2]
            rewards.append(-1)
            dones=[False, False]
            if v[ind0][0][2] is not None:
                self.states[agent] = (
                    tuple(v[ind0][0][0]),
                    v[ind0][0][1], 
                    tuple(v[ind0][0][2])
                )
            else:
                self.states[agent] = (
                    tuple(v[ind0][0][0]),
                    v[ind0][0][1], 
                    None
                )
        self.renderer.render_step()

        return observations, rewards, dones, info


    def get_state(self, agent):
        return self.states[agent]


    def render(self):
        return self.renderer.get_renders()

    def _render_frame(self, mode: str):
        # This will be the function called by the renderer to collect a single frame
        assert mode is not None

        import pygame # avoid global pygame dependency

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255)) # make a white canvas
        pix_square_size = (
            # divide the canvas into square sizes corresponding to the grid
            # clearly here it is easier if the grid is square
            self.window_size / self.xsize
        )

        # For each of the feed locations draw a green square
        for floc in self.warehouse_api.feedpoints:
            pygame.draw.rect(
                canvas,
                (0, 255 ,0),
                pygame.Rect(
                    pix_square_size * np.array(floc, dtype=np.int64),
                    (pix_square_size, pix_square_size)
                ),
            )
        

        # if an agent is carrying a rack draw this rack first
        for ag in range(self.warehouse_api.nagents):
            if self.agent_rack_positions[ag] is not None:
                arack = self.agent_rack_positions[ag]
                pygame.draw.rect(
                    canvas,
                    (0, 0 ,255),
                    pygame.Rect(
                        pix_square_size * np.array(arack, dtype=np.int64) + 0.1 * pix_square_size,
                        (pix_square_size - 0.2 * pix_square_size, pix_square_size - 0.2 * pix_square_size)
                    ),
                )
        for r in self.warehouse_api.racks:
            pygame.draw.rect(
                canvas,
                (0, 0 ,255),
                pygame.Rect(
                    pix_square_size * np.array(r, dtype=np.int64) + 0.1 * pix_square_size,
                    (pix_square_size - 0.2 * pix_square_size, pix_square_size - 0.2 * pix_square_size)
                ),
            )
        for ag in range(self.warehouse_api.nagents):
            if self.orig_rack_positions[ag] is not None:
                if not all(np.equal(self.agent_rack_positions[ag],self.orig_rack_positions[ag])):
                    r = self.orig_rack_positions[ag]
                    pygame.draw.rect(
                        canvas,
                        (255, 255 ,255),
                        pygame.Rect(
                            pix_square_size * np.array(r, dtype=np.int64) + 0.1 * pix_square_size,
                            (pix_square_size - 0.2 * pix_square_size, pix_square_size - 0.2 * pix_square_size)
                        ),
                    )
                    
        
        # draw each of the agents as a circle
        for ag in range(self.warehouse_api.nagents):
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                (np.array(self.agent_obs[ag]["a"], dtype=np.int64) + 0.5) * pix_square_size,
                pix_square_size / 3
            )

        # Add some grid lines
        for x in range(self.xsize + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )
        if mode == "human":
            assert self.window is not None
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array or single_rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            

    

        

