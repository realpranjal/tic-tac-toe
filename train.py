import random
import json # Moved import to top (Best practice)

Q = {}

# ### CHANGED: Hyperparameters
# Alpha 0.3 -> 0.1 (Stability)
# Gamma 0.6 -> 0.9 (Foresight)
alpha = 0.1  
gamma = 0.9
eps = 1.0

# ### CHANGED: Added current_eps argument so we use the decayed value
def choose_action(state, actions, current_eps):
    # ### CHANGED: usage of global 'eps' -> local 'current_eps'
    if random.random() < current_eps:
        return random.choice(actions)
    return max(actions, key=lambda a: Q.get((state, a), 0))

# ### DELETED: The 'update' function was removed entirely. 
# In Q-learning, updates happen inside the game loop, not in a separate function.

def empty_board():
    return [0] * 9

def legal_moves(board):
    return [i for i in range(9) if board[i] == 0]

def play(board, move, player):
    b = board[:]
    b[move] = player
    return b

wins = [
    (0,1,2),(3,4,5),(6,7,8),
    (0,3,6),(1,4,7),(2,5,8),
    (0,4,8),(2,4,6)
]

def check_win(board):
    for a,b,c in wins:
        s = board[a] + board[b] + board[c]
        if s == 3:  return 1
        if s == -3: return -1
    if 0 not in board: return 0
    return None

def encode(board):
    return tuple(board)

# ### CHANGED: Added current_eps argument
def play_game(current_eps):
    board = empty_board()
    # ### DELETED: history = [] (We don't need history for Q-learning)
    
    player = 1
    
    # ### NEW: Track previous state to update it after opponent moves
    prev_state = None
    prev_action = None
    
    while True:
        state = encode(board)
        actions = legal_moves(board)
        
        # ### CHANGED: Pass current_eps
        action = choose_action(state, actions, current_eps)
        
        # ### DELETED: history.append(...)
        
        # Make the move
        next_board = play(board, action, player)
        result = check_win(next_board)
        
        # ### NEW: Logic to handle Q-update continuously
        # 1. Define Reward and Done flag
        if result is None:
            reward = 0
            done = False
        elif result == 0: # Draw
            reward = 0
            done = True
        else: # Win/Loss
            reward = 1 
            done = True

        # ### NEW: Update the PREVIOUS player's move (The Opponent)
        # If I (current player) am in a good state, the opponent's last move was BAD.
        if prev_state is not None:
            if done:
                # Game over: The value is just the inverted reward
                target = -1 * reward
            else:
                # Game continues: The value is inverted best future move
                best_current_val = max([Q.get((state, a), 0) for a in actions], default=0)
                target = -1 * best_current_val

            # Update Formula
            old_q = Q.get((prev_state, prev_action), 0)
            # Note: We use global 'alpha' here which is now dynamic in the main loop
            Q[(prev_state, prev_action)] = old_q + alpha * ((gamma * target) - old_q)

        # ### NEW: If game is done, update the CURRENT player's move too
        if done:
            old_q = Q.get((state, action), 0)
            Q[(state, action)] = old_q + alpha * (reward - old_q)
            break

        # ### NEW: Store current state as 'previous' for next iteration
        prev_state = state
        prev_action = action
        
        # ### CHANGED: Update board variable name (board -> next_board logic is implicit)
        board = next_board 
        player *= -1

# --- Main Loop ---

# ### CHANGED: Reduced N (1mil -> 500k is sufficient with correct logic)
N = 500_000 
min_eps = 0.0
min_alpha = 0.01 # ### NEW: Floor for alpha

for i in range(1, N + 1):
    # ### CHANGED: Epsilon decay formula slightly smoother
    curr_eps = max(min_eps, 1.0 * (0.99999 ** i))
    
    # ### NEW: Alpha Decay (Starts at 0.5, decays to 0.01)
    # This was missing in your code, causing unstable learning at the end.
    alpha = max(min_alpha, 0.5 * (0.99999 ** i))
    
    # ### CHANGED: Pass curr_eps to function
    play_game(curr_eps)

    if i % 50000 == 0:
        print(
            f"Training: {i}/{N} | "
            f"eps={curr_eps:.4f} | "
            f"alpha={alpha:.4f} | " # ### NEW: Printing Alpha
            f"Q size={len(Q)}"
        )

# JSON saving remains exactly the same
Q_json = {}
for (state, action), value in Q.items():
    state_str = str(list(state))
    key = f"{state_str}_{action}"
    Q_json[key] = value

with open("q.json", "w") as f:
    json.dump(Q_json, f)

print(f"Saved {len(Q_json)} Q-values to q.json")