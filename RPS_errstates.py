# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

import random
import numpy as np


"""
reference:
https://github.com/raul1991/rock-paper-scissors-RL

result
--------- you vs quincy ----------
Final results: {'p1': 393, 'p2': 10, 'tie': 597}
Player 1 win rate: 97.51861042183623%
--------- you vs abbey ----------
Final results: {'p1': 501, 'p2': 498, 'tie': 1}
Player 1 win rate: 50.150150150150154%
--------- you vs kris ----------
Final results: {'p1': 0, 'p2': 0, 'tie': 1000}
Player 1 win rate: 0%
--------- you vs mrugesh ----------
Final results: {'p1': 841, 'p2': 159, 'tie': 0}
Player 1 win rate: 84.1%


"""

class Bot(object):
    # our states can be either "ROCK, PAPER or SCISSORS"
    state_space = 3

    # three actions by our player
    action_space = 3

    q_table = np.random.uniform(low=-2, high=5, size=(3, 3))
    total_reward, reward = 0, 0
    avg_rewards_list = []
    avg_reward = 0
    result = 'DRAW'
    tags = ["R", "P", "S"]
    # looses to map
    loses_to = {
        "0": 1,  # rock loses to paper
        "1": 2,  # paper loses to scissor
        "2": 0  # scissor loses to rock
    }

    def __init__(self, alpha=0.5, gamma=0.2, epsilon=0.8, min_eps=0, episodes=1000, verbose=False):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_eps = min_eps
        self.episodes = episodes
        # Calculate episodic reduction in epsilon
        self.reduction = (epsilon - min_eps) / episodes

        self.verbose = verbose

    # either explore or exploit, any which ways return the next action
    def bot_move(self, player_move):
        action = 0
        # Determine next action - epsilon greedy strategy
        if np.random.random() < 1 - self.epsilon:
            if self.verbose:
                print("Exploiting....")

            action = np.argmax(self.q_table[player_move])
        else:
            if self.verbose:
                print("Exploring.....")

            action = np.random.randint(0, self.action_space)

        # Decay epsilon
        if self.epsilon > self.min_eps:
            self.epsilon -= self.reduction

        if self.verbose:
            print("choose ", self.tags[action])

        return action

    def get_action(self, player_move):
        if self.verbose:
            print("Get action....")

        action = np.argmax(self.q_table[player_move])

        return action

    def get_reward(self, player, bot):
        reward = 0

        if self.get_result(player, bot) == 'WIN':
            reward = 5
        elif self.get_result(player, bot) == 'LOSE':
            reward = -2
        else:
            # Draw case
            reward = 4

        return reward

    # update q_table
    def update_experience(self, state, action, reward):
        delta = self.alpha * (reward + self.gamma * np.max(self.q_table[action]) - self.q_table[state, action])
        self.q_table[state, action] += delta

    def print_stats(self, player, bot, reward):
        if self.verbose:
            print("Player move : {0}, bot: {1}, reward: {2}, result: {3}, total_reward: {4}".format(self.tags[player],
                                                                                                    self.tags[bot], reward,
                                                                                                    self.result,
                                                                                                    self.total_reward))
            print(self.q_table)

    # returns either a WIN, LOSE or a DRAW to indicate the same.
    def get_result(self, player_move, bot_move):
        if bot_move == player_move:
            self.result = 'DRAW'
        elif self.loses_to[str(bot_move)] == player_move:
            self.result = 'LOSE'
        else:
            self.result = 'WIN'

        return self.result

    def get_avg_rewards(self):
        return self.avg_rewards_list

    def learn(self, player_move):
        # add reward
        bot_move = self.bot_move(player_move)
        reward = self.get_reward(player_move, bot_move)
        self.total_reward += reward
        self.avg_rewards_list.append(reward)
        # update experience
        self.update_experience(player_move, bot_move, reward)
        self.print_stats(player_move, bot_move, reward)


# when each opponent start, the opponent_history will be a empty list,
# At that moment , we should create a new bot to learn that opponents' rules

bot_player = None


def player(prev_play, opponent_history=[], verbose=False):
    # print("call player")
    # print(prev_play)
    # print(len(opponent_history))

    global bot_player

    play_list = ["R", "P", "S"]
    win_dict = {"R": "P", "P": "S", "S": "R"}

    if len(opponent_history) == 0:
        bot_player = Bot(verbose=False)

    # suppose opponent's play is R, before real first round
    prev_play_index = 0

    if prev_play in play_list:
        opponent_history.append(prev_play)
        prev_play_index = play_list.index(prev_play)

    bot_player.learn(prev_play_index)

    me_play_index = bot_player.get_action(prev_play_index)

    if verbose:
        print(f"opponent most possible next play is {me_play_index}")

    me_play = play_list[me_play_index]

    return me_play
