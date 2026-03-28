# models.py
from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List, Optional
from enum import Enum

class ActionType(str, Enum):
    IDLE = "idle"
    TRACK = "track"
    MANEUVER = "maneuver"

class OrbitalAction(Action):
    """Action for the OrbitalOps environment."""
    action_type: ActionType = Field(..., description="Must be 'idle', 'track', or 'maneuver'.")
    station_id: Optional[str] = Field(None, description="Required if 'track'.")
    dv_x: Optional[float] = Field(None, description="Delta-V in X. Required if 'maneuver'.")
    dv_y: Optional[float] = Field(None, description="Delta-V in Y. Required if 'maneuver'.")

class OrbitalObservation(Observation):
    """Observation from the OrbitalOps environment."""
    # Note: 'reward', 'done', and 'metadata' are already built into the Observation base class!
    time_step: int = Field(default=0, description="Current simulation step.")
    position_x: float = Field(default=0.0, description="X coordinate in km.")
    position_y: float = Field(default=0.0, description="Y coordinate in km.")
    velocity_x: float = Field(default=0.0, description="X velocity in km/s.")
    velocity_y: float = Field(default=0.0, description="Y velocity in km/s.")
    fuel_remaining: float = Field(default=100.0, description="Fuel remaining (0.0 to 100.0).")
    positional_uncertainty: float = Field(default=5.0, description="State uncertainty in km.")
    visible_stations: List[str] = Field(default_factory=list, description="Visible stations.")