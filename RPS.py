# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

import random
import numpy as np


"""
reference:
https://github.com/raul1991/rock-paper-scissors-RL

state should be opponent's play, last N plays is better
https://stats.stackexchange.com/questions/291906/can-reinforcement-learning-be-stateless

Note: this code implement last one state, but for abbey and kris, the wining rate is not improved appearantly.
but in most times, it can beat all players.

--------- you vs quincy ----------
Final results: {'p1': 386, 'p2': 140, 'tie': 474}
Player 1 win rate: 73.38403041825094%
--------- you vs abbey ----------
Final results: {'p1': 3525, 'p2': 3306, 'tie': 3169}
Player 1 win rate: 51.60298638559509%
--------- you vs kris ----------
Final results: {'p1': 3295, 'p2': 3262, 'tie': 3443}
Player 1 win rate: 50.251639469269485%
--------- you vs mrugesh ----------
Final results: {'p1': 609, 'p2': 230, 'tie': 161}
Player 1 win rate: 72.58641239570917%


state can also be designed as WIN LOSE TIE
https://github.com/dennylslee/rock-paper-scissors-DeepRL

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
    def update_experience(self, state, action, reward, player_next_move):
        reward_next_move = np.max(self.q_table[player_next_move])
        delta = self.alpha * (reward + self.gamma * reward_next_move - self.q_table[state, action])
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

    def learn(self, player_move, bot_move, player_next_move):
        # add reward
        reward = self.get_reward(player_move, bot_move)

        self.total_reward += reward
        self.avg_rewards_list.append(reward)

        # update experience
        self.update_experience(player_move, bot_move, reward, player_next_move)
        self.print_stats(player_move, bot_move, reward)


# when each opponent start, the opponent_history will be a empty list,
# At that moment , we should create a new bot to learn that opponents' rules

bot_player = None


def player(opponent_prev_play, opponent_history, me_prev_play, me_history, num_games, verbose=False):
    # print("call player")
    # print(prev_play)
    # print(len(opponent_history))

    global bot_player

    play_list = ["R", "P", "S"]
    win_dict = {"R": "P", "P": "S", "S": "R"}

    if len(opponent_history) == 0:
        bot_player = Bot(verbose=verbose, episodes=num_games)

    # suppose opponent's play is R, before real first round
    opponent_prev_play_index = 0

    if opponent_prev_play in play_list:
        if len(opponent_history) > 0:
            opponent_prev_prev_play = opponent_history[-1]
            opponent_prev_prev_play_index = play_list.index(opponent_prev_prev_play)

        opponent_history.append(opponent_prev_play)
        opponent_prev_play_index = play_list.index(opponent_prev_play)

    if me_prev_play in play_list:
        if len(me_history) > 1:
            me_prev_prev_play = me_history[-1]
            me_prev_prev_play_index = play_list.index(me_prev_prev_play)

        me_history.append(me_prev_play)

    if len(opponent_history) >= 3:
        state = opponent_prev_prev_play_index
        next_state = opponent_prev_play_index
        action = me_prev_prev_play_index
        bot_player.learn(state, action, next_state)

    me_play_index = bot_player.bot_move(opponent_prev_play_index)

    if verbose:
        print(f"opponent most possible next play is {me_play_index}")

    me_play = play_list[me_play_index]

    return me_play
