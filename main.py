import gymnasium as gym
from gymnasium.utils.play import play


def main():
    # LunarLander-v3 con render in RGB per play()
    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    # Azioni discrete:
    # 0 = niente
    # 1 = motore sinistro
    # 2 = motore principale
    # 3 = motore destro
    keys_to_action = {
        "": 0,    # nessun tasto
        "w": 2,   # motore principale (su)
        "a": 1,   # motore sinistro
        "d": 3,   # motore destro
        "wa": 2,  # se tieni W+A, tengo il principale
        "wd": 2,  # W+D idem
    }

    play(
        env,
        keys_to_action=keys_to_action,
        noop=0,
        fps=30,
    )

    env.close()


if __name__ == "__main__":
    main()
