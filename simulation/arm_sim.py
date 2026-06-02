# Responsible for the physical simulation of the 2-joint robotic arm and all pygame rendering.

import numpy as np
import math
import pygame

class ArmSimulation:

    # ── display constants ──────────────────────────────────────────
    WIDTH, HEIGHT = 600, 600          # pixel dimensions of the pygame window
    BG_COLOR      = (15,  15,  25)    # background colour (dark navy)
    ARM1_COLOR    = (80, 160, 255)    # colour of the first arm segment (blue)
    ARM2_COLOR    = (50, 220, 180)    # colour of the second arm segment (teal)
    JOINT_COLOR   = (255, 255, 255)   # colour of the joint circles (white)
    TARGET_COLOR  = (255,  80,  80)   # colour of the target dot (red)
    EE_COLOR      = (255, 220,  50)   # colour of the end-effector dot (yellow)

    def __init__(self, render=False):
        # ── arm geometry ───────────────────────────────────────────
        self.length1 = 120            # pixel length of the first arm segment
        self.length2 = 120            # pixel length of the second arm segment
        self.joint1_angle = 0.0       # current angle of joint 1 in degrees
        self.joint2_angle = 0.0       # current angle of joint 2 relative to joint 1 in degrees

        # ── rendering ─────────────────────────────────────────────
        self.render_mode = render             # whether to draw the arm to the screen each step
        self.screen      = None               # pygame display surface (None until _init_pygame is called)
        self.clock       = None               # pygame clock used to cap the frame rate
        self.target      = np.array([0.0, 0.0])   # target position in sim-space coordinates (set by env)

        if self.render_mode:
            self._init_pygame()

    # Initialises the pygame window, display surface, and clock.
    def _init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Double-Joint Arm — RL")
        self.clock  = pygame.time.Clock()

    # Computes the (x, y) positions of joint 1 and the end-effector using forward kinematics.
    def calculate_positions(self):
        """Returns (joint1_pos, end_effector) in sim-space (origin = centre)."""
        x1 = self.length1 * math.cos(math.radians(self.joint1_angle))   # x component of joint 1 in sim-space
        y1 = self.length1 * math.sin(math.radians(self.joint1_angle))   # y component of joint 1 in sim-space
        joint1_pos = np.array([x1, y1])                                  # (x, y) position of joint 1 in sim-space

        total_angle = self.joint1_angle + self.joint2_angle              # absolute angle of the second segment in degrees
        x2 = joint1_pos[0] + self.length2 * math.cos(math.radians(total_angle))  # x component of end-effector in sim-space
        y2 = joint1_pos[1] + self.length2 * math.sin(math.radians(total_angle))  # y component of end-effector in sim-space
        end_effector = np.array([x2, y2])                                # (x, y) position of the end-effector in sim-space

        return joint1_pos, end_effector

    # Applies a (delta_j1, delta_j2) angle increment to each joint, wrapping both to [-180, 180].
    def step(self, action):
        """
        action: array-like of shape (2,)
            action[0] → delta degrees for joint 1
            action[1] → delta degrees for joint 2
        Angles are clamped to [-180, 180].
        """
        self.joint1_angle = (self.joint1_angle + action[0] + 180) % 360 - 180   # new joint 1 angle wrapped to [-180, 180]
        self.joint2_angle = (self.joint2_angle + action[1] + 180) % 360 - 180   # new joint 2 angle wrapped to [-180, 180]

    # Resets both joints to 0°, returning the arm to its default straight-right pose.
    def reset(self):
        self.joint1_angle = 0.0   # joint 1 angle reset to 0 degrees
        self.joint2_angle = 0.0   # joint 2 angle reset to 0 degrees

    # Converts a sim-space (x, y) coordinate to a pygame screen pixel coordinate, flipping the y-axis.
    def _sim_to_screen(self, pos):
        """Flip y-axis and offset to screen centre."""
        cx, cy = self.WIDTH // 2, self.HEIGHT // 2   # pixel coordinates of the screen centre
        return int(cx + pos[0]), int(cy - pos[1])    # y is flipped because pygame y grows downward

    # Draws one frame: background, crosshair grid, target, arm segments, joints, and end-effector.
    def render(self):
        if not self.render_mode or self.screen is None:
            return

        # ── event pump (keeps window responsive) ──────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill(self.BG_COLOR)

        origin = np.array([0.0, 0.0])   # sim-space position of the shoulder (fixed base)
        j1, ee = self.calculate_positions()

        o_scr  = self._sim_to_screen(origin)        # screen pixel position of the shoulder origin
        j1_scr = self._sim_to_screen(j1)            # screen pixel position of joint 1
        ee_scr = self._sim_to_screen(ee)            # screen pixel position of the end-effector
        tg_scr = self._sim_to_screen(self.target)   # screen pixel position of the target

        # grid cross-hair
        pygame.draw.line(self.screen, (30, 30, 45),
                         (self.WIDTH//2, 0), (self.WIDTH//2, self.HEIGHT), 1)
        pygame.draw.line(self.screen, (30, 30, 45),
                         (0, self.HEIGHT//2), (self.WIDTH, self.HEIGHT//2), 1)

        # target
        pygame.draw.circle(self.screen, self.TARGET_COLOR, tg_scr, 10)
        pygame.draw.circle(self.screen, (255, 130, 130), tg_scr, 14, 2)

        # arm segments
        pygame.draw.line(self.screen, self.ARM1_COLOR, o_scr,  j1_scr, 6)
        pygame.draw.line(self.screen, self.ARM2_COLOR, j1_scr, ee_scr, 6)

        # joints
        pygame.draw.circle(self.screen, self.JOINT_COLOR, o_scr,  7)
        pygame.draw.circle(self.screen, self.JOINT_COLOR, j1_scr, 7)

        # end-effector
        pygame.draw.circle(self.screen, self.EE_COLOR, ee_scr, 9)

        pygame.display.flip()
        self.clock.tick(60)

    # Shuts down pygame and clears the screen reference so the window is destroyed cleanly.
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
