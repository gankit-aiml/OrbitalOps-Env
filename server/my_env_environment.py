# server/my_env_environment.py
import math
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import OrbitalAction, OrbitalObservation
except (ImportError, ModuleNotFoundError):
    from models import OrbitalAction, OrbitalObservation

# --- Physics Constants ---
MU = 3.986e5
R_EARTH = 6371.0
MAX_RADIUS = 100000.0
DT = 120.0
OMEGA_E = 7.292e-5
MAX_STEPS = 50

TASKS = {
    "Task_1_Easy": {"px": 7500.0, "py": 0.0, "vx": 0.0, "vy": 7.29, "fuel": 100.0, "uncertainty": 5.0},
    "Task_2_Medium": {"px": 20000.0, "py": 0.0, "vx": 0.0, "vy": 3.0, "fuel": 100.0, "uncertainty": 10.0},
    "Task_3_Hard": {"px": 0.0, "py": 8000.0, "vx": -6.5, "vy": 0.0, "fuel": 5.0, "uncertainty": 90.0}
}

def grade_task(history, task_name) -> float:
    if not history: return 0.0
    if task_name == "Task_1_Easy":
        if len(history) < 200: return 0.0
        avg_uncert = sum(s.positional_uncertainty for s in history) / len(history)
        score = 1.0 - (avg_uncert / 100.0)
        if history[-1].fuel_remaining < 100.0: score -= 0.5
        return max(0.0, min(1.0, score))
    elif task_name == "Task_2_Medium":
        visible_steps = sum(1 for s in history if len(s.visible_stations) > 0)
        score = visible_steps / 80.0
        if history[-1].positional_uncertainty > 100.0 or len(history) < 200: score *= 0.5
        return max(0.0, min(1.0, score))
    else: # Task_3_Hard
        return max(0.0, min(1.0, len(history) / 200.0))

class MyEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = False # Keep to False so we can track the global state easily

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.history = []
        self.current_task = "Task_1_Easy"
        self.obs_data = {} # Holds the physics state

    def reset(self, task_name="Task_1_Easy") -> OrbitalObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.current_task = task_name if task_name in TASKS else "Task_1_Easy"
        config = TASKS[self.current_task]
        
        self.history = []
        self.obs_data = {
            "time_step": 0, "position_x": config["px"], "position_y": config["py"],
            "velocity_x": config["vx"], "velocity_y": config["vy"],
            "fuel_remaining": config["fuel"], "positional_uncertainty": config["uncertainty"],
            "visible_stations": []
        }
        self._update_visibility()
        
        obs = OrbitalObservation(**self.obs_data, reward=0.0, done=False)
        self.history.append(obs)
        return obs

    def _rk4(self):
        x, y = self.obs_data["position_x"], self.obs_data["position_y"]
        vx, vy = self.obs_data["velocity_x"], self.obs_data["velocity_y"]
        
        def f(s):
            r = math.sqrt(s[0]**2 + s[1]**2)
            return [s[2], s[3], -MU*s[0]/r**3, -MU*s[1]/r**3]
        
        k1 = f([x, y, vx, vy])
        k2 = f([x + 0.5*DT*k1[0], y + 0.5*DT*k1[1], vx + 0.5*DT*k1[2], vy + 0.5*DT*k1[3]])
        k3 = f([x + 0.5*DT*k2[0], y + 0.5*DT*k2[1], vx + 0.5*DT*k2[2], vy + 0.5*DT*k2[3]])
        k4 = f([x + DT*k3[0], y + DT*k3[1], vx + DT*k3[2], vy + DT*k3[3]])

        self.obs_data["position_x"] += (DT/6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        self.obs_data["position_y"] += (DT/6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        self.obs_data["velocity_x"] += (DT/6.0) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        self.obs_data["velocity_y"] += (DT/6.0) * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])

    def _update_visibility(self):
        t = self.obs_data["time_step"] * DT
        visible = []
        for i, offset in enumerate([0, math.pi*0.66, math.pi*1.33]):
            theta = (OMEGA_E * t) + offset
            sx, sy = R_EARTH * math.cos(theta), R_EARTH * math.sin(theta)
            if ((self.obs_data["position_x"] - sx)*sx + (self.obs_data["position_y"] - sy)*sy) >= 0:
                visible.append(f"Station_{i+1}")
        self.obs_data["visible_stations"] = visible

    def step(self, action: OrbitalAction) -> OrbitalObservation:
        self._state.step_count += 1
        reward_val, done, info = 0.0, False, "Nominal"
        
        # Action Logic
        if action.action_type == "track" and action.station_id in self.obs_data["visible_stations"]:
            self.obs_data["positional_uncertainty"] = 1.0
            reward_val += 1.0
        elif action.action_type == "maneuver":
            dv = math.sqrt((action.dv_x or 0)**2 + (action.dv_y or 0)**2)
            if self.obs_data["fuel_remaining"] >= dv:
                self.obs_data["velocity_x"] += (action.dv_x or 0)
                self.obs_data["velocity_y"] += (action.dv_y or 0)
                self.obs_data["fuel_remaining"] -= dv
                self.obs_data["positional_uncertainty"] += (2.0 + dv * 10)
                reward_val -= 1.0
        
        # Physics
        self._rk4()
        self.obs_data["positional_uncertainty"] += 0.2
        self.obs_data["time_step"] += 1
        self._update_visibility()

        # Terminations
        r = math.sqrt(self.obs_data["position_x"]**2 + self.obs_data["position_y"]**2)
        if r < R_EARTH or r > MAX_RADIUS or self.obs_data["positional_uncertainty"] > 100:
            done, reward_val, info = True, -10.0, "Failure"
        elif self.obs_data["time_step"] >= MAX_STEPS:
            done = True

        obs = OrbitalObservation(**self.obs_data, reward=reward_val, done=done, metadata={"info": info})
        self.history.append(obs)
        return obs

    @property
    def state(self) -> State:
        return self._state