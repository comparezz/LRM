import random, time, argparse
from agents.run_lrm import run_lrm_experiments
from agents.run_baseline import run_baseline_experiments
from agents.learning_parameters import LearningParameters
from worlds.grid_world import GridWorldParams
from worlds.game import GameParams

def set_environment(env, lp):
    """
    Returns the parameters of the selected environment. 
    It also set an adequate number of training steps (which depends on the environment's difficulty) 
    """
    movement_noise = 0.05
    if env == "cookie_world":
        game_type = "cookieworld"
        file_map = "../maps/cookie.txt"
        lp.max_learning_steps = int(3e6)
        lp.train_steps = int(3e6)
    if env == "symbol_world":
        game_type = "symbolworld"
        file_map = "../maps/symbol.txt"
        lp.max_learning_steps = int(2e6)
        lp.train_steps = int(2e6)
    if env == "keys_world":
        game_type = "keysworld"
        file_map = "../maps/2-keys.txt"
        lp.max_learning_steps = int(4e6)
        lp.train_steps = int(4e6)
    return GameParams(GridWorldParams(game_type, file_map, movement_noise))


def run_lrm_agent(rl, env, n_seed, n_workers):

    save = True
    print("Running", rl, "in", env, "using seed", n_seed)
    if save: print("SAVING RESULTS!")
    else:    print("*NOT* SAVING RESULTS!")

    # Setting the learning parameters
    lp = LearningParameters()

    lp.set_rm_learning(rm_init_steps=200e3, rm_u_max=10, rm_preprocess=True, rm_tabu_size=10000, 
                       rm_lr_steps=100, rm_workers=n_workers)
    lp.set_rl_parameters(gamma=0.9, train_steps=None, episode_horizon=int(5e3), epsilon=0.1, max_learning_steps=None)
    lp.set_test_parameters(test_freq = int(1e4))
    lp.set_deep_rl(lr = 5e-5, learning_starts = 50000, train_freq = 1, target_network_update_freq = 100, 
                    buffer_size = 100000, batch_size = 32, use_double_dqn = True, num_hidden_layers = 5, num_neurons = 64)

    # Setting the environment
    env_params = set_environment(env, lp)

    # Choosing the RL algorithm
    print("\n----------------------")
    print("LRM agent:", rl)
    print("----------------------\n")

    # Running the experiment
    run_lrm_experiments(env_params, lp, rl, n_seed, save)

def run_baseline_agent(rl, env, n_seed, k_order):
    save = True
    print("Running", str(k_order)+"-order", rl, "in", env, "using seed", n_seed)
    if save: print("SAVING RESULTS!")
    else:    print("*NOT* SAVING RESULTS!")

    # Setting the learning parameters
    lp = LearningParameters()

    lp.set_rl_parameters(gamma=0.9, train_steps=int(5e6), episode_horizon=int(5e3), epsilon=0.1, max_learning_steps=None)
    lp.set_test_parameters(test_freq = int(1e4))
    lp.set_deep_rl(lr = 5e-5, learning_starts = 50000, train_freq = 1, target_network_update_freq = 100, 
                    buffer_size = 100000, batch_size = 32, use_double_dqn = True, num_hidden_layers = 5, num_neurons = 64)

    # Setting the environment
    env_params = set_environment(env, lp)

    # Choosing the RL algorithm
    print("\n----------------------")
    print("Baseline:", rl)
    print("----------------------\n")

    # Running the experiment
    run_baseline_experiments(env_params, lp, rl, k_order, n_seed, save)


if __name__ == "__main__":

    # EXAMPLE: python3 run.py --agent="lrm-dqn" --world="cookie" --seed=1 --workers=16
    # EXAMPLE: python3 run.py --agent="lrm-dqn" --world="keys" --seed=1 --workers=16
    # EXAMPLE: python3 run.py --agent="lrm-dqn" --world="symbol" --seed=1 --workers=16

    # Getting params
    agents = ["lrm-dqn", "lrm-qrm", "dqn", "human"]
    worlds = ["keys", "cookie", "symbol"]

    parser = argparse.ArgumentParser(prog="run_experiments", description='Runs the selected RL agent over the selected world.')
    parser.add_argument('--agent', default='lrm-dqn', type=str, 
                        help='This parameter indicates which RL algorithm to use. The options are: ' + str(agents))
    parser.add_argument('--world', default='cookie', type=str, 
                        help='This parameter indicates which world to solve. The options are: ' + str(worlds))
    parser.add_argument('--seed', default=0, type=int, 
                        help='This parameter indicates which random seed to use.')
    parser.add_argument('--workers', default=16, type=int, 
                        help='This parameter indicates the number of threads that will be used when learning the RM.')


    args = parser.parse_args()
    assert args.agent in agents, "Agent " + args.algorithm + " hasn't been implemented yet"
    assert args.world in worlds, "World " + args.world + " hasn't been defined yet"
    assert args.workers > 0, "The number of workers must be greater than zero"

    # Running the experiment
    rl     = args.agent
    env    = args.world + "_world"
    n_seed = args.seed
    n_workers = args.workers

    if rl.startswith("lrm-"):
        # LRM
        rl = rl.replace("lrm-","")
        run_lrm_agent(rl, env, n_seed, n_workers)
    else:
        # K-Order baseline
        k_order = 10 if rl == "dqn" else 1 # size of the fixed memory used by the DQN baseline
        run_baseline_agent(rl, env, n_seed, k_order)
