# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

import random
from hmmlearn import hmm
import numpy as np
import math

states = ["0", "1", "2", "3", "4"]
'''
--------- you vs quincy ----------
Final results: {'p1': 443, 'p2': 242, 'tie': 315}
Player 1 win rate: 64.67153284671532%
--------- you vs abbey ----------
Final results: {'p1': 317, 'p2': 376, 'tie': 307}
Player 1 win rate: 45.74314574314574%
--------- you vs kris ----------
Final results: {'p1': 322, 'p2': 390, 'tie': 288}
Player 1 win rate: 45.2247191011236%
--------- you vs mrugesh ----------
Final results: {'p1': 722, 'p2': 218, 'tie': 60}
Player 1 win rate: 76.80851063829788%
'''

# states = ["0", "1", "2", "3"]
'''
--------- you vs quincy ----------
Final results: {'p1': 421, 'p2': 292, 'tie': 287}
Player 1 win rate: 59.04628330995793%
--------- you vs abbey ----------
Final results: {'p1': 312, 'p2': 362, 'tie': 326}
Player 1 win rate: 46.29080118694362%
--------- you vs kris ----------
Final results: {'p1': 322, 'p2': 398, 'tie': 280}
Player 1 win rate: 44.72222222222222%
--------- you vs mrugesh ----------
Final results: {'p1': 503, 'p2': 207, 'tie': 290}
Player 1 win rate: 70.84507042253522%
'''

# states = ["0", "1", "2"]
'''
--------- you vs quincy ----------
Final results: {'p1': 391, 'p2': 239, 'tie': 370}
Player 1 win rate: 62.06349206349206%
--------- you vs abbey ----------
Final results: {'p1': 309, 'p2': 397, 'tie': 294}
Player 1 win rate: 43.76770538243626%
--------- you vs kris ----------
Final results: {'p1': 313, 'p2': 405, 'tie': 282}
Player 1 win rate: 43.5933147632312%
--------- you vs mrugesh ----------
Final results: {'p1': 624, 'p2': 204, 'tie': 172}
Player 1 win rate: 75.36231884057972%
'''

# states = ["0", "1"]
'''
--------- you vs quincy ----------
Final results: {'p1': 331, 'p2': 237, 'tie': 432}
Player 1 win rate: 58.27464788732394%
--------- you vs abbey ----------
Final results: {'p1': 327, 'p2': 422, 'tie': 251}
Player 1 win rate: 43.65821094793058%
--------- you vs kris ----------
Final results: {'p1': 310, 'p2': 458, 'tie': 232}
Player 1 win rate: 40.36458333333333%
--------- you vs mrugesh ----------
Final results: {'p1': 524, 'p2': 391, 'tie': 85}
Player 1 win rate: 57.267759562841526%
'''

n_states = len(states)


observations_dict = {
                        0: "R",
                        1: "P",
                        2: "S"
                    }
n_features = len(observations_dict)


def player(prev_play, opponent_history=[], verbose=False):
    # print("call player")
    # print(prev_play)
    # print(len(opponent_history))

    global n_states

    play_list = ["R", "P", "S"]
    win_dict = {"R": "P", "P": "S", "S": "R"}

    if prev_play in play_list:
        opponent_history.append(prev_play)

    # default
    me_play = random.choice(play_list)

    learning_point = 40

    look_back = 4

    # look_back = 4
    '''
--------- you vs quincy ----------
Final results: {'p1': 740, 'p2': 100, 'tie': 160}
Player 1 win rate: 88.09523809523809%
--------- you vs abbey ----------
Final results: {'p1': 417, 'p2': 385, 'tie': 198}
Player 1 win rate: 51.99501246882793%
--------- you vs kris ----------
Final results: {'p1': 411, 'p2': 417, 'tie': 172}
Player 1 win rate: 49.63768115942029%
--------- you vs mrugesh ----------
Final results: {'p1': 797, 'p2': 179, 'tie': 24}
Player 1 win rate: 81.65983606557377%
    '''

    # look_back = 3
    '''
--------- you vs quincy ----------
Final results: {'p1': 776, 'p2': 88, 'tie': 136}
Player 1 win rate: 89.81481481481481%
--------- you vs abbey ----------
Final results: {'p1': 359, 'p2': 401, 'tie': 240}
Player 1 win rate: 47.23684210526316%
--------- you vs kris ----------
Final results: {'p1': 331, 'p2': 416, 'tie': 253}
Player 1 win rate: 44.310575635876845%
--------- you vs mrugesh ----------
Final results: {'p1': 801, 'p2': 169, 'tie': 30}
Player 1 win rate: 82.57731958762886%
    '''

    if len(opponent_history) > learning_point:
        if verbose:
            print("now enter learn and predict mode")
            print(f"enter learn stage, with learning window {learning_point}")

        # observations = opponent_history[-learning_point:]
        observations = opponent_history[:]
        observations = [[play_list.index(x)] for x in observations]
        observations = np.array(observations)

        model = hmm.MultinomialHMM(n_components=n_states,
                                   n_iter=100,
                                   tol=1,
                                   verbose=False,
                                   init_params="ste")

        model_trained = model.fit(observations)

        start = model_trained.startprob_
        if verbose:
            print("-------- start ---------")
            print(start)

        transition = model_trained.transmat_
        if verbose:
            print("-------- transition ---------")
            print(transition)

        emission = model_trained.emissionprob_
        if verbose:
            print("-------- emission ---------")
            print(emission)

        if verbose:
            print(f"enter predict stage, with look back {look_back}")

        obs_now = opponent_history[-look_back:]
        obs_now = "".join(obs_now)

        # print(obs_now)
        options = [obs_now + v for v in play_list]

        options_prob = [0, 0, 0]
        for i, one_option in enumerate(options):
            one_option = list(one_option)
            one_option = [[play_list.index(x)] for x in one_option]
            one_option = np.array(one_option)
            one_prob = model_trained.score(one_option)
            options_prob[i] = one_prob

            if verbose:
                print(f"possible option {one_option} with probability {one_prob}")

        options_prob = np.array(options_prob)

        best_index = np.argmax(options_prob)
        best_play = play_list[best_index]

        if verbose:
            print(f"opponent most possible next play is {best_play}")

        me_play = win_dict[best_play]

    return me_play
