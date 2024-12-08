import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import pygame
class SprayPaintingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config, reward_config, render_mode=None):
        self._config = config

        # Grid parameters
        width = config['environment']['grid_params']['grid_size'][1]
        height = config['environment']['grid_params']['grid_size'][0]
        self._resolution = config['environment']['grid_params']['resolution']

        self._grid_width = int(width / self._resolution)
        self._grid_height = int(height / self._resolution)

        self._x_sections = config['environment']['grid_params']['x_sections']
        self._y_sections = config['environment']['grid_params']['y_sections']

        # Observation limits
        self._x_min = -config['environment']['limit_params']['x_max']
        self._y_min = -config['environment']['limit_params']['y_max']

        self._x_max = width + config['environment']['limit_params']['x_max']
        self._y_max = height + config['environment']['limit_params']['y_max']
        self._z_max = config['environment']['limit_params']['z_max']

        self._max_angles = config['environment']['limit_params']['max_angles']
        self._max_aperture = config['environment']['limit_params']['max_aperture']

        # Action limits
        self._max_move = config['environment']['limit_params']['max_move']
        self._max_rotate = config['environment']['limit_params']['max_rot']

        self._coverage_threshold = config['environment']['grid_params']['coverage_threshold']

        self._deposit_rate = config['environment']['grid_params']['deposit_rate'] # amount deposited per time step (between 0 and 1)

        # self.observation_space = spaces.Dict({
        #     "position": spaces.Box(low=np.array([self._x_min, self._y_min, 0.0]), high=np.array([self._x_max, self._y_max, self._z_max]), shape=(3,), dtype=np.float64),
        #     "angles": spaces.Box(low=-np.array(self._max_angles), high=np.array(self._max_angles), shape=(2,), dtype=np.float64),
        #     "aperture": spaces.Box(low=0.0, high=self._max_aperture, shape=(1,), dtype=np.float64),
        #     "surface_grid": spaces.Box(low=0, high=1, shape=(self._grid_width * self._grid_height,), dtype=np.float64)
        # })

        self.observation_space = spaces.Dict({
            "position": spaces.Box(low=np.array([0.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), shape=(3,), dtype=np.float64),
            "angles": spaces.Box(low=-np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), shape=(2,), dtype=np.float64),
            "aperture": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64),
            "section_coverage_averages": spaces.Box(low=0, high=1, shape=(self._x_sections * self._y_sections,), dtype=np.float64)
        })

        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1]),  # Min limits for the actions, normalized to [-1, 1]
            high=np.array([1, 1, 1, 1, 1, 1]),  # Max limits for the actions, normalized to [-1, 1]
            shape=(6,),  
            dtype=np.float32  # Change dtype to np.float32 to avoid casting errors
        )

        # States
        self._position = np.array([-1, -1, -1])
        self._angles = np.array([-1, -1])
        self._aperture = 0.0
        self._surface_grid = np.zeros((self._grid_width, self._grid_height))
        
        # Reward parameters
        self._k1 = reward_config['weights']['time_cost']
        self._k2 = reward_config['weights']['usage_cost']
        self._k3 = reward_config['weights']['progress_reward']
        self._k4 = reward_config['weights']['coverage_reward']

        # Rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # For human-mode rendering
        self.window = None
        self.clock = None

    def _get_obs(self):
        position_normalized = (self._position - np.array([self._x_min, self._y_min, 0])) / np.array([self._x_max - self._x_min, self._y_max - self._y_min, self._z_max])
        angles_normalized = (self._angles + np.array(self._max_angles)) / (2 * np.array(self._max_angles))
        aperture_normalized = self._aperture / self._max_aperture  # Normalize aperture

        # Define the number of sections for averaging coverage (e.g., 4 sections per axis)
        num_sections_x = self._x_sections  # Divide grid into 4 sections along the x-axis
        num_sections_y = self._y_sections # Divide grid into 4 sections along the y-axis
        
        # Get the height and width of the grid
        height, width = self._surface_grid.shape
        
        # Calculate the section size along each axis
        section_height = height // num_sections_y
        section_width = width // num_sections_x
        
        # Initialize a list to hold the average coverage values for each section
        section_coverage_averages = []
        
        # Loop through each section and calculate the average coverage
        for i in range(num_sections_y):
            for j in range(num_sections_x):
                # Calculate the row and column indices for the current section
                row_start = i * section_height
                row_end = (i + 1) * section_height
                col_start = j * section_width
                col_end = (j + 1) * section_width
                
                # Extract the section from the grid
                section = self._surface_grid[row_start : row_end, col_start : col_end]
                
                # Calculate the average coverage for the current section
                section_coverage_averages.append(np.mean(section >= self._coverage_threshold))
        
        return {
            "position": position_normalized, # [x, y, z]
            "angles": angles_normalized, # [elevation, azimuth]
            "aperture": np.array([aperture_normalized]),
            "section_coverage_averages": np.array(section_coverage_averages)
        }

    def _get_info(self):
        return { "average_spray_saturation": np.mean(self._surface_grid),
                 "max_saturation": np.max(self._surface_grid),
                 "min_saturation": np.min(self._surface_grid),
                 "coverage_percentage": 100 * np.mean(self._surface_grid >= self._coverage_threshold) }

    def reset(self, seed:Optional[int]=None, options: Optional[dict]=None):
        super().reset(seed=seed)

        # Choose start states at random
        self._position = np.array([
            np.random.uniform(low=self._x_min, high=self._x_max),
            np.random.uniform(low=self._y_min, high=self._y_max),
            np.random.uniform(low=0, high=self._z_max)         
        ])

        self._angles = np.array([
            np.random.uniform(low=-self._max_angles[0], high=self._max_angles[0]), # 
            np.random.uniform(low=-self._max_angles[1], high=self._max_angles[1]),
        ])

        observation = self._get_obs()

        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        move_action = action[:3] * self._max_move  # Scale to real move limits
        rotate_action = action[3:5] * self._max_rotate  # Scale to real rotate limits
        adjust_aperture_action = (action[5] + 1) / 2 * self._max_aperture # Scale to real aperture limits
        
        self._position = np.clip(
            self._position + move_action,
            np.array([self._x_min, self._y_min, 0.0]),
            np.array([self._x_max, self._y_max, self._z_max])
        )

        self._angles = np.clip(
            self._angles + rotate_action,
            -np.array(self._max_angles),
            np.array(self._max_angles)
        )

        self._aperture = np.clip(
            self._aperture + adjust_aperture_action, 
            0, 
            self._max_aperture
        )

        # Apply spray to grid
        self._prev_grid = self._surface_grid
        self._apply_spray()

        terminated = self._check_done()
        truncated = False
        reward = self._calculate_reward()
        observation = self._get_obs()

        info = self._get_info()

        # Add individual rewards to info dictionary
        info.update(reward)
        
        if self.render_mode == "human":
            self._render_frame()

        return observation, sum(reward.values()), terminated, truncated, info

    @property
    def _axis_intersect(self):
        """
        Calculate intersection point of spray point axis with surface
        """
        x = self._position[0] + self._position[2] * np.tan(np.deg2rad(self._angles[1]))
        y = self._position[1] - self._position[2] * np.tan(np.deg2rad(self._angles[0]))

        return (x, y)
    
    @property
    def _ellipse_section(self):
        """
        Compute the ellipse
        """
        # Calculate orientation
        projection = np.array([self._position[2] * np.tan(np.deg2rad(self._angles[1])),
                               -self._position[2] * np.tan(np.deg2rad(self._angles[0]))])

        orientation = np.arctan2(projection[1], projection[0]) # orientation in rad

        L = np.linalg.norm(np.array([self._position[0], self._position[1]]) - self._axis_intersect)
        new_intersect = np.array([self._position[0] + L, self._position[1]])
        adj = np.sqrt(self._position[2] ** 2 + L ** 2)

        # Compute first axis
        angle = np.arctan2(L, self._position[2])

        x_rot_max = self._position[0] + self._position[2] * np.tan(np.deg2rad(self._aperture / 2) + angle)
        x_rot_min = self._position[0] + self._position[2] * np.tan(angle - np.deg2rad(self._aperture / 2))
        
        axis1 = abs(x_rot_max - x_rot_min)

        # Compute second axis
        y_rot_max = self._position[1] - adj * np.tan(np.deg2rad(self._aperture / 2))
        y_rot_min = self._position[1] + adj * np.tan(np.deg2rad(self._aperture / 2))

        axis2 = abs(y_rot_min - y_rot_max)

        R = np.array([[np.cos(orientation), -np.sin(orientation)],
                      [np.sin(orientation), np.cos(orientation)]])

        xc = (x_rot_max + x_rot_min) / 2
        yc = (y_rot_max + y_rot_min) / 2
        
        rel_x = xc - self._position[0]
        rel_y = yc - self._position[1]

        rotated_center = R @ np.array([rel_x, rel_y])

        center = rotated_center + np.array([self._position[0], self._position[1]])
        
        x0 = center[0]
        y0 = center[1]

        a = max(axis1, axis2)
        b = min(axis1, axis2)

        return x0, y0, a, b, orientation

    @property
    def _conic_section(self):
        """
        Return the conic section made by the spray cone on the surface
        """

        if np.isclose(self._angles[0], 0, atol=1e-5) and np.isclose(self._angles[1], 0, atol=1e-5):
            radius = self._position[2] * np.tan(np.deg2rad(self._aperture / 2))
            return {
                "type": "circle",
                "center": self._axis_intersect,
                "radius": radius
            }
        else:
            x_c, y_c, a, b, orientation = self._ellipse_section

            return {
                "type": "ellipse",
                "center": (x_c, y_c),
                "major_axis": a,
                "minor_axis": b,
                "orientation": orientation
            }

    def _apply_spray(self):
        equation = self._conic_section
        resolution = self._resolution

        if equation["type"] == "circle":
            x_c, y_c = equation["center"]
            radius = equation["radius"] 
            area = np.pi * (radius ** 2)
            paint_amount = self._deposit_rate / area

            x_min = max(0, x_c - radius)
            x_max = min(self._grid_width, x_c + radius)
            y_min = max(0, y_c - radius)
            y_max = min(self._grid_height, y_c + radius)

            col_min = int(x_min / resolution)
            col_max = int(x_max / resolution)
            row_min = int(y_min / resolution)
            row_max = int(y_max / resolution)
        
            for col in range(col_min, col_max):
                for row in range(row_min, row_max):
                    x = (col + 0.5) * resolution
                    y = (row + 0.5) * resolution

                    dx = (x - x_c)
                    dy = (y - y_c)

                    if dx ** 2 + dy ** 2 <= radius ** 2:
                        if row < self._surface_grid.shape[0] and col < self._surface_grid.shape[1]:
                            c = int(x / resolution)
                            r = int(y / resolution)
                            self._surface_grid[row, col] += paint_amount * resolution ** 2
        else:
            x_c, y_c = equation["center"]
            a = equation["major_axis"] / (2 * resolution)
            b = equation["minor_axis"] / (2 * resolution)
            orientation = equation["orientation"]

            area = np.pi * equation["major_axis"] * equation["minor_axis"]
            paint_amount = self._deposit_rate / area

            cos_theta = np.cos(orientation)
            sin_theta = np.sin(orientation)

            x_min = int(max(0, (x_c - a) / resolution))
            x_max = int(min(self._grid_width, (x_c + a) / resolution))
            y_min = int(max(0, (y_c - b) / resolution))
            y_max = int(min(self._grid_height, (y_c + b) / resolution))

            for x in range(x_min, x_max):
                for y in range(y_min, y_max):
                    dx = x * resolution - x_c
                    dy = y * resolution - y_c
                    ellipse_x = dx * cos_theta + dy * sin_theta
                    ellipse_y = -dx * sin_theta + dy * cos_theta

                    # Check if the cell lies within the ellipse
                    if (ellipse_x / (a * resolution)) ** 2 + (ellipse_y / (b * resolution)) ** 2 <= 1 + 1e-5:
                        self._surface_grid[y, x] += paint_amount * resolution ** 2

        # Clip
        self._surface_grid = np.clip(
            self._surface_grid,
            0,
            1
        )

    def _calculate_reward(self):
        time_cost = -1 # Cost for time step
        
        usage_cost = -self._aperture # Cost for using paint

        # Progress_reward: reward coverage
        progress_reward = np.mean(self._surface_grid) - np.mean(self._prev_grid)

        # Coverage reward: reward for increasing coverage percentage
        coverage_reward = np.sum(self._surface_grid >= self._coverage_threshold) / (self._grid_width * self._grid_height)

        # Completion bonus: reward for full coverage
        completion_bonus = 100 if np.all(self._surface_grid >= self._coverage_threshold) else 0

        # Return as a dictionary of rewards
        return {
                'time_cost': self._k1 * time_cost,
                'usage_cost': self._k2 * usage_cost,
                'progress_reward': self._k3 * progress_reward,
                'coverage_reward': self._k4 * coverage_reward,
                'completion_bonus': completion_bonus
                }
    
    def _check_done(self):
        return bool(np.all(self._surface_grid >= self._coverage_threshold))

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window_size = 600
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

  
        # Calculate the buffer and grid rendering parameters
        buffer_fraction = 0.1  # Buffer as a fraction of the total canvas size
        buffer_x = buffer_fraction * self.window_size
        buffer_y = buffer_fraction * self.window_size

        grid_width = self._grid_width
        grid_height = self._grid_height

        cell_width = (self.window_size - 2 * buffer_x) / grid_width
        cell_height = (self.window_size - 2 * buffer_y) / grid_height
        agent_x = self._position[0]  # Get the agent's x position
        agent_y = self._position[1]  # Get the agent's y position
        start_x = buffer_x
        start_y = buffer_y
        # Convert the agent's position to canvas coordinates
        # agent_canvas_x = start_x + agent_x * cell_width
        # agent_canvas_y = start_y + agent_y * cell_height
        # Convert the agent's position from global coordinates to canvas coordinates
        agent_canvas_x = buffer_x + (self._position[0] - self._x_min) / (self._x_max - self._x_min) * (self.window_size - 2 * buffer_x)
        agent_canvas_y = buffer_y + (self._position[1] - self._y_min) / (self._y_max - self._y_min) * (self.window_size - 2 * buffer_y)

        pygame.draw.circle(canvas, (255, 0, 0), (int(agent_canvas_x), int(agent_canvas_y)), 5)
        
        agent_int_x = buffer_x + (self._axis_intersect[0] - self._x_min) / (self._x_max - self._x_min) * (self.window_size - 2 * buffer_x)
        agent_int_y = buffer_y + (self._axis_intersect[1] - self._y_min) / (self._y_max - self._y_min) * (self.window_size - 2 * buffer_y)

        pygame.draw.circle(canvas, (255, 0, 0), (int(agent_int_x), int(agent_int_y)), 2)

        # Draw the surface rectangle
        surface_rect = pygame.Rect(buffer_x, buffer_y,
                                   self.window_size - 2 * buffer_x,
                                   self.window_size - 2 * buffer_y)
        
        pygame.draw.rect(canvas, (0, 0, 0), surface_rect, 3)

        for x in range(grid_width):
            for y in range(grid_height):
                # Compute the cell's alpha value based on the saturation
                saturation = self._surface_grid[y, x]  # y, x due to how numpy arrays are indexed
                alpha = int(saturation * 255)  # Scale saturation to 0-255 for alpha

                # Define the cell rectangle on the canvas
                rect_x = buffer_x + x * cell_width
                rect_y = buffer_y + y * cell_height
                cell_rect = pygame.Rect(rect_x, rect_y, cell_width, cell_height)

                # Fill the cell with a color and alpha
                cell_color = (0, 0, 255, alpha)  # Blue with varying transparency
                cell_surface = pygame.Surface((cell_width, cell_height), pygame.SRCALPHA)
                cell_surface.fill(cell_color)

                # Blit the cell surface onto the canvas
                canvas.blit(cell_surface, cell_rect.topleft)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()