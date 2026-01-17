def categorize_episode_outcome(obs, episode_steps, last_step_reward):
    """
    Categories:
      - time_limit
      - out_of_bounds
      - landed_success
      - crashed

    Rules:
      0) time_limit if episode_steps >= 1000
      1) out_of_bounds if abs(x) > 1.0
      2) landed_success if final step reward contains the +100 landing bonus
      3) crashed if final step reward contains the -100 crash penalty
      4) out_of_bounds otherwise
    """
    x, y, vel_x, vel_y, angle, angular_vel, left_leg, right_leg = obs

    # 0) Time limit by steps (you wanted this deterministic rule)
    if episode_steps >= 1000:
        return "time_limit"

    # 1) Out of viewport rule
    if abs(x) >= 1.0 or y > 2.0:
        return "out_of_bounds"

    # 2-3) Terminal bonus/penalty detection (thresholds, not exact equality)
    if last_step_reward >= 90.0:
        return "landed_success"
    if last_step_reward <= -90.0:
        return "crashed"



def get_outcome_icon(outcome):
    icons = {
        "landed_success": "🟢",
        "crashed": "🔴",
        "out_of_bounds": "🟡",
        "time_limit": "🔵",
    }
    return icons.get(outcome, "❓")
