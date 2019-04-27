12345678
87654321
*****
Comments:
our evaluation function combines several heuristics:
1. snake heuristic - in general, we would like our tiles to be ordered in a zigzag snakelike shape, sorted from lowest
 to highest number. this is helpful because it makes merging tiles easy, due to the fact that closer tiles usually have
 closer numbers. in addition it allows combos of consecutive merges. in order to make this happen, we created a snake
 like scores matrix and calculated a state's score by entry-entry multiplication of the heuristic matrix and the board
  state.
2.empty tiles heuristic - when playing 2048 we always want to have as little tiles as possible on the board, because
  when the board is full the game is over. using the snake heuristic only can cause a situation when we avoid a possible
  merge. this is why we decided to reward a state according to the number of empty tiles it has.
3. merge tiles heuristic - another aspect that was not considered in the previous heuristics was that a state is better
 if it allows more merges in the future. due to that we decided to add another heuristic that rewards states for
 containing possible future merges.

we tried different combinations of these heuristics and other heuristics that we decided to omit (multiplying by
matrices other than the snake matrix, rewarding states for having the top score tile in one of the corners etc.) and
took the weights that gave us the best scores.
