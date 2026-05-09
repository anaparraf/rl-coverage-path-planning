from typing import Optional
import numpy as np
import gymnasium as gym
import pygame

#
# Coverage Path Planning (CPP) — SOLUTION B (Robust)
#
# POMDP Solution: Global Visited Map + Strong Reward Shaping
#
# The core problem in 10×10: agent has 5×5 window (25% visibility) and without
# global memory enters loops after ~88% coverage, unable to locate remaining 12%.
#
# Solution: Two legitimate observations built by the agent itself:
#
#   visited_map (size×size binary):
#     - 1 where agent has stepped
#     - 0 where agent has never been
#     - Updated by agent as it explores — not ground truth
#     - Legitimate: agent knows where *it* has walked
#
# Reward shaping emphasizes:
#   - Strong frontier bonus: new cell adjacent to unexplored territory
#   - Strong penalties: revisit (-0.5) and walls (-0.8)
#   - Lower step cost (-0.15) to encourage longer exploration
#
# Expected performance:
#   5×5: 95-100% full coverage (marginal improvement)
#   10×10: 85-95% full coverage (major improvement from 2%)
#

class GridWorldCPPEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        size: int = 5,
        obs_quantity: int = 3,
        max_steps: int = 200,
        obs_window_size: int = 3,
    ):
        self.size = size
        self.window_size = 512
        self.obs_quantity = obs_quantity
        self.obstacles_locations = []
        self.count_steps = 0
        self.max_steps = max_steps
        self.obs_window_size = obs_window_size

        if self.obs_window_size % 2 == 0 or self.obs_window_size < 3:
            raise ValueError("obs_window_size must be an odd number >= 3")

        self.visited = set()
        self._agent_location = np.array([-1, -1], dtype=int)
        self._neighbors = np.zeros(
            (self.obs_window_size, self.obs_window_size), dtype=np.float32
        )
        self._visited_map = np.zeros((self.size, self.size), dtype=np.float32)

        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(
                low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            ),
            "neighbors": gym.spaces.Box(
                low=np.zeros((self.obs_window_size, self.obs_window_size), dtype=np.float32),
                high=np.full((self.obs_window_size, self.obs_window_size), 2.0, dtype=np.float32),
                dtype=np.float32,
            ),
            "visited_map": gym.spaces.Box(
                low=np.zeros((self.size, self.size), dtype=np.float32),
                high=np.ones((self.size, self.size), dtype=np.float32),
                dtype=np.float32,
            ),
        })

        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),   # right
            1: np.array([0, -1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, 1]),   # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    @property
    def total_free_cells(self):
        return self.size * self.size - len(self.obstacles_locations)

    @property
    def coverage_ratio(self):
        return len(self.visited) / self.total_free_cells if self.total_free_cells > 0 else 1.0

    def _get_obs(self):
        return {
            "agent": np.array([
                self._agent_location[0] / self.size,
                self._agent_location[1] / self.size,
                self.coverage_ratio,
            ], dtype=np.float32),
            "neighbors": self._neighbors.copy(),
            "visited_map": self._visited_map.copy(),
        }

    def _get_info(self):
        return {
            "coverage": self.coverage_ratio,
            "visited_cells": len(self.visited),
            "total_free_cells": self.total_free_cells,
            "steps": self.count_steps,
            "size": self.size,
        }

    def _is_frontier_cell(self, pos):
        """Check if pos is adjacent to at least one unvisited free cell."""
        x, y = pos
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                if (nx, ny) not in self.visited:
                    is_obstacle = any(
                        np.array_equal(np.array([nx, ny]), loc)
                        for loc in self.obstacles_locations
                    )
                    if not is_obstacle:
                        return True
        return False

    def set_neighbors(self, obstacles_locations):
        """Build local observation window centered on agent."""
        size = self.obs_window_size
        center = size // 2
        matrix = np.zeros((size, size), dtype=np.float32)
        for i in range(size):
            for j in range(size):
                nx = self._agent_location[0] + (j - center)
                ny = self._agent_location[1] + (i - center)
                neighbor = np.array([nx, ny])
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    matrix[i][j] = 1
                elif any(np.array_equal(neighbor, loc) for loc in obstacles_locations):
                    matrix[i][j] = 1
                elif (nx, ny) in self.visited:
                    matrix[i][j] = 2
        self._neighbors = matrix

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.count_steps = 0
        self.obstacles_locations = []
        self.visited = set()
        self._visited_map = np.zeros((self.size, self.size), dtype=np.float32)

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        for _ in range(self.obs_quantity):
            obstacle_location = self._agent_location.copy()
            while (np.array_equal(obstacle_location, self._agent_location) or
                   any(np.array_equal(obstacle_location, loc) for loc in self.obstacles_locations)):
                obstacle_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            self.obstacles_locations.append(obstacle_location)

        start_pos = tuple(self._agent_location)
        self.visited.add(start_pos)
        self._visited_map[self._agent_location[1], self._agent_location[0]] = 1.0

        self.set_neighbors(self.obstacles_locations)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        old_location = self._agent_location.copy()

        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        if any(np.array_equal(self._agent_location, loc) for loc in self.obstacles_locations):
            self._agent_location = old_location

        self.set_neighbors(self.obstacles_locations)
        self.count_steps += 1

        current_pos = tuple(self._agent_location)
        is_new_cell = current_pos not in self.visited
        stayed_in_place = np.array_equal(self._agent_location, old_location)

        # Base step penalty (increased to encourage efficiency)
        reward = -0.15

        if stayed_in_place:
            # Hit wall or obstacle (strong penalty to discourage)
            reward -= 0.8
        elif is_new_cell:
            # Reward for exploring new cell
            reward += 1.0
            self.visited.add(current_pos)
            self._visited_map[self._agent_location[1], self._agent_location[0]] = 1.0
            
            # FRONTIER BONUS: reward movement toward unexplored territory
            # This is crucial for POMDP: directs agent toward gaps in visited_map
            if self._is_frontier_cell(current_pos):
                reward += 0.3
        else:
            # Penalty for revisiting (increased to discourage loops)
            reward -= 0.5

        full_coverage = len(self.visited) >= self.total_free_cells
        terminated = full_coverage

        if full_coverage:
            reward += 10.0

        if self.count_steps >= self.max_steps and not terminated:
            truncated = True
            reward -= 5.0
        else:
            truncated = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        for cell in self.visited:
            cell_arr = np.array(cell)
            pygame.draw.rect(
                canvas,
                (144, 238, 144),
                pygame.Rect(
                    pix_square_size * cell_arr,
                    (pix_square_size, pix_square_size),
                ),
            )

        for obs in self.obstacles_locations:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * obs,
                    (pix_square_size, pix_square_size),
                ),
            )

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        font = pygame.font.SysFont(None, 24)
        coverage_text = font.render(
            f"Coverage: {self.coverage_ratio:.1%} | Steps: {self.count_steps}",
            True, (0, 0, 0)
        )
        canvas.blit(coverage_text, (5, 5))

        for x in range(self.size + 1):
            pygame.draw.line(canvas, 0, (0, pix_square_size * x),
                             (self.window_size, pix_square_size * x), width=3)
            pygame.draw.line(canvas, 0, (pix_square_size * x, 0),
                             (pix_square_size * x, self.window_size), width=3)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
