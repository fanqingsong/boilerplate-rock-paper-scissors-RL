### Assignment

For this challenge, you will create a program to play Rock, Paper, Scissors. A program that picks at random will usually win 50% of the time. To pass this challenge your program must play matches against four different bots, winning at least 60% of the games in each match.

In the file `RPS.py` you are provided with a function called `player`. The function takes an argument that is a string describing the last move of the opponent ("R", "P", or "S"). The function should return a string representing the next move for it to play ("R", "P", or "S").

A player function will receive an empty string as an argument for the first game in a match since there is no previous play.

The file `RPS.py` shows an example function that you will need to update. The example function is defined with two arguments (`player(prev_play, opponent_history = [])`). The function is never called with a second argument so that one is completely optional. The reason why the example function contains a second argument (`opponent_history = []`) is because that is the only way to save state between consecutive calls of the `player` function. You only need the `opponent_history` argument if you want to keep track of the opponent_history.

*Hint: To defeat all four opponents, your program may need to have multiple strategies that change depending on the plays of the opponent.*

### Development

Do not modify `RPS_game.py`. Write all your code in `RPS.py`. For development, you can use `main.py` to test your code. 

`main.py` imports the game function and bots from `RPS_game.py`.

To test your code, play a game with the `play` function. The `play` function takes four arguments:
- two players to play against each other (the players are actually functions)
- the number of games to play in the match
- an optional argument to see a log of each game. Set it to `True` to see these messages.

```py
play(player1, player2, num_games[, verbose])
```
For example, here is how you would call the function if you want `player` and `quincy` to play 1000 games against each other and you want to see the results of each game:
```py
play(player, quincy, 1000, verbose=True)
```

Click the "run" button and `main.py` will run.

### Testing 

The unit tests for this project are in `test_module.py`. We imported the tests from `test_module.py` to `main.py` for your convenience. If you uncomment the last line in `main.py`, the tests will run automatically whenever you hit the "run" button.

### Submitting

Copy your project's URL and submit it to freeCodeCamp.


### result

```

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


```

### 
