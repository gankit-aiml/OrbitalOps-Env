# inference.py
import os
import json
from typing import Dict
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# Import our environment and schemas
from models import OrbitalAction, ActionType
from server.my_env_environment import MyEnvironment, TASKS, grade_task

# ==========================================
# MANDATORY VARIABLES (Fixed the typo from their example)
# ==========================================
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "no_key_provided"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"

MAX_STEPS = 50
TEMPERATURE = 0.2

def heuristic_fallback(obs_dict: dict) -> OrbitalAction:
    """If the API fails (404, rate limit), this ensures we don't crash."""
    if obs_dict.get("visible_stations"):
        return OrbitalAction(action_type=ActionType.TRACK, station_id=obs_dict["visible_stations"][0])
    if obs_dict.get("positional_uncertainty", 0) > 80:
        return OrbitalAction(action_type=ActionType.MANEUVER, dv_x=-0.05, dv_y=0.05)
    return OrbitalAction(action_type=ActionType.IDLE)

# Circuit-breaker: if auth fails once, skip all further API calls
_api_disabled = False

def get_action_from_llm(client: OpenAI, obs_dict: dict) -> OrbitalAction:
    global _api_disabled
    if _api_disabled:
        return heuristic_fallback(obs_dict)

    prompt = (
        f"Spacecraft State: {json.dumps(obs_dict)}. "
        "Goal: Keep uncertainty low (<10km). "
        "If a station is in 'visible_stations', call the 'track' tool. "
        "If no stations are visible, call 'idle'. "
        "If uncertainty is critical (>80km), call 'maneuver' to fix your orbit."
    )

    tools = [{
        "type": "function",
        "function": {
            "name": "execute_action",
            "description": "Execute spacecraft control",
            "parameters": OrbitalAction.model_json_schema()
        }
    }]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            tool_choice="auto",
            temperature=TEMPERATURE
        )
        
        if response.choices and response.choices[0].message.tool_calls:
            args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            return OrbitalAction(**args)
            
    except Exception as e:
        err = str(e)
        if "401" in err or "402" in err or "403" in err:
            print(f"LLM API Error: {e}.")
            print("[WARNING] Auth error detected - disabling API calls for this run. Using heuristic fallback for all remaining steps.")
            _api_disabled = True
        else:
            print(f"LLM API Error: {e}. Using fallback action.")
        
    return heuristic_fallback(obs_dict)

def run_inference() -> Dict[str, float]:
    print(f"[INFO] Connecting to {API_BASE_URL} using model {MODEL_NAME}")
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    env = MyEnvironment()
    scores = {}

    for task_name in TASKS.keys():
        print(f"\n--- Starting {task_name} ---")
        state = env.reset(task_name)
        
        for step in range(1, MAX_STEPS + 1):
            action = get_action_from_llm(client, state.model_dump())
            state = env.step(action)
            done = state.done
            info = state.metadata.get("info", "Nominal") if state.metadata else "Nominal"
            
            if done:
                print(f"Episode complete at step {step}. Info: {info}")
                break
                
        score = round(grade_task(env.history, task_name), 3)
        scores[task_name] = score
        print(f"Score for {task_name}: {score}")
        
    return scores

if __name__ == "__main__":
    final_scores = run_inference()
    print("\n[RESULTS] Final Baseline Scores:")
    print(json.dumps(final_scores, indent=4))