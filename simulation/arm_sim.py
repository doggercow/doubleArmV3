import numpy as np
import math
import pygame

class ArmSimulation:

    # ── display constants ──────────────────────────────────────────
    WIDTH, HEIGHT = 600, 600
    BG_COLOR      = (15,  15,  25)
    ARM1_COLOR    = (80, 160, 255)
    ARM2_COLOR    = (50, 220, 180)
    JOINT_COLOR   = (255, 255, 255)
    TARGET_COLOR  = (255,  80,  80)
    EE_COLOR      = (255, 220,  50)

    def __init__(self, render=False):
        # ── arm geometry ───────────────────────────────────────────
        self.length1 = 120
        self.length2 = 120
        self.joint1_angle = 0.0   # degrees
        self.joint2_angle = 0.0   # degrees

        # ── rendering ─────────────────────────────────────────────
        self.render_mode = render
        self.screen      = None
        self.clock       = None
        self.target      = np.array([0.0, 0.0])   # set by env

        if self.render_mode:
            self._init_pygame()

    # ── pygame setup ──────────────────────────────────────────────
    def _init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Double-Joint Arm — RL")
        self.clock  = pygame.time.Clock()

    # ── core math (your original logic, untouched) ────────────────
    def calculate_positions(self):
        """Returns (joint1_pos, end_effector) in sim-space (origin = centre)."""
        x1 = self.length1 * math.cos(math.radians(self.joint1_angle))
        y1 = self.length1 * math.sin(math.radians(self.joint1_angle))
        joint1_pos = np.array([x1, y1])

        total_angle = self.joint1_angle + self.joint2_angle
        x2 = joint1_pos[0] + self.length2 * math.cos(math.radians(total_angle))
        y2 = joint1_pos[1] + self.length2 * math.sin(math.radians(total_angle))
        end_effector = np.array([x2, y2])

        return joint1_pos, end_effector

    # ── action interface (called by env) ──────────────────────────
    def step(self, action):
        """
        action: array-like of shape (2,)
            action[0] → delta degrees for joint 1
            action[1] → delta degrees for joint 2
        Angles are clamped to [-180, 180].
        """
        self.joint1_angle = float(np.clip(
            self.joint1_angle + action[0], -180, 180))
        self.joint2_angle = float(np.clip(
            self.joint2_angle + action[1], -180, 180))

    # ── reset ─────────────────────────────────────────────────────
    def reset(self):
        self.joint1_angle = 0.0
        self.joint2_angle = 0.0

    # ── rendering ─────────────────────────────────────────────────
    def _sim_to_screen(self, pos):
        """Flip y-axis and offset to screen centre."""
        cx, cy = self.WIDTH // 2, self.HEIGHT // 2
        return int(cx + pos[0]), int(cy - pos[1])   # y flipped for screen coords

    def render(self):
        if not self.render_mode or self.screen is None:
            return

        # ── event pump (keeps window responsive) ──────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill(self.BG_COLOR)

        origin     = np.array([0.0, 0.0])
        j1, ee     = self.calculate_positions()

        o_scr  = self._sim_to_screen(origin)
        j1_scr = self._sim_to_screen(j1)
        ee_scr = self._sim_to_screen(ee)
        tg_scr = self._sim_to_screen(self.target)

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

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None