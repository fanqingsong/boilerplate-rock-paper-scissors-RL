# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

import random
import mchmm as mc
import numpy as np
import math


def to_log_val(prob):
    trans_prob = 1 + prob

    log_val = math.log(trans_prob)

    return log_val


def calc_prob(hsi, one_option, machine):
    prob = 0

    all_obs = machine.observations
    all_obs = all_obs.tolist()

    obs_seq = list(one_option)

    start_state_index = hsi[0]
    start_obs = obs_seq[0]
    start_obs_index = all_obs.index(start_obs)

    start_state_prob = machine.pi[start_state_index]
    start_state_prob = to_log_val(start_state_prob)

    start_obs_prob = machine.ep[start_state_index][start_obs_index]

    prob += start_state_prob + start_obs_prob

    for i in range(len(hsi) - 1):
        cur_index = hsi[i]
        next_index = hsi[i + 1]

        trans_prob = machine.tp[cur_index][next_index]

        prob += to_log_val(trans_prob)

        next_obs = obs_seq[i+1]
        next_obs_index = all_obs.index(next_obs)
        next_obs_prob = machine.ep[next_index][next_obs_index]

        prob += to_log_val(next_obs_prob)

    return prob


def player(prev_play, opponent_history=[], verbose=False):
    # print("call player")
    # print(prev_play)
    # print(len(opponent_history))

    play_list = ["R", "P", "S"]
    win_dict = {"R": "P", "P": "S", "S": "R"}

    if prev_play in play_list:
        opponent_history.append(prev_play)

    # default
    me_play = random.choice(play_list)

    learning_window = 100

    look_back = 2

    if len(opponent_history) > learning_window:
        if verbose:
            print("now enter learn and predict mode")
            print(f"enter learn stage, with learning window {learning_window}")

        obs_seq = opponent_history[-learning_windowRPS.py:]

        machine = mc.HiddenMarkovModel().from_baum_welch(obs_seq, states=['0', '1', '2'], obs=np.array(play_list))

        if verbose:
            print(f"enter predict stage, with look back {look_back}")

        obs_now = opponent_history[-look_back:]
        obs_now = "".join(obs_now)

        # print(obs_now)
        options = [obs_now + v for v in play_list]

        options_prob = [0, 0, 0]
        for i, one_option in enumerate(options):
            hs, hsi = machine.viterbi(one_option)
            one_prob = calc_prob(hsi, one_option, machine)
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
