# server/app.py
from fastapi import HTTPException
from pydantic import BaseModel
import sys
import os

# Add parent directory to path for imports when needed
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install dependencies.") from e

try:
    from ..models import OrbitalAction, OrbitalObservation
    from .my_env_environment import MyEnvironment, TASKS, grade_task
except (ImportError, ModuleNotFoundError):
    try:
        from models import OrbitalAction, OrbitalObservation
        from my_env_environment import MyEnvironment, TASKS, grade_task
    except (ImportError, ModuleNotFoundError):
        from my_env.models import OrbitalAction, OrbitalObservation
        from my_env.server.my_env_environment import MyEnvironment, TASKS, grade_task

# Create the official app
app = create_app(
    MyEnvironment,
    OrbitalAction,
    OrbitalObservation,
    env_name="my_env",
    max_concurrent_envs=1, 
)

# --- INJECT HACKATHON REQUIRED ENDPOINTS ---

@app.get("/tasks")
def get_tasks():
    """Hackathon Requirement: Returns list of tasks and the action schema."""
    return {
        "tasks": list(TASKS.keys()),
        "action_schema": OrbitalAction.model_json_schema()
    }

@app.get("/grader")
def get_grader():
    """Hackathon Requirement: Returns grader score."""
    # Since max_concurrent_envs=1, we can grab the active environment from the server's env_manager
    env_instance = app.state.env_manager.get_env("default") 
    if not env_instance or not hasattr(env_instance, "history") or not env_instance.history:
        return {"score": 0.0}
    
    score = grade_task(env_instance.history, env_instance.current_task)
    return {"score": score}

@app.post("/baseline")
def run_baseline():
    """Hackathon Requirement: Trigger inference script."""
    try:
        import sys
        import os
        # Add parent directory to path for imports
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import inference
        scores = inference.run_inference()
        return {"baseline_scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()