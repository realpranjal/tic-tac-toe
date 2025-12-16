import random
import json

Q = {}

# --- HYPERPARAMETERS ---
WIN_REWARD = 100.0
LOSS_REWARD = -100.0
DRAW_REWARD = 0.0
gamma = 0.8  

def legal_moves(board):
    return [i for i in range(9) if board[i] == 0]

def check_win(board):
    wins = [
        (0,1,2), (3,4,5), (6,7,8),
        (0,3,6), (1,4,7), (2,5,8),
        (0,4,8), (2,4,6)
    ]
    for a, b, c in wins:
        s = board[a] + board[b] + board[c]
        if s == 3:  return 1   # 1 wins
        if s == -3: return -1  # -1 wins
    if 0 not in board: return 0 # Draw
    return None

# --- CRITICAL: CANONICAL STATE ---
# This converts the board so the "Current Player" is ALWAYS '1'.
# If I am O (-1), I flip the board so my pieces look like '1'.
def get_canonical_state(board, player):
    if player == 1:
        return tuple(board)
    else:
        # Flip perspective: -1 -> 1, 1 -> -1, 0 -> 0
        return tuple([-x for x in board])

def choose_action(board, player, current_eps):
    state = get_canonical_state(board, player)
    actions = legal_moves(board)
    
    if random.random() < current_eps:
        return random.choice(actions)
    
    random.shuffle(actions)
    # Note: We look up 'state' which is already relative to 'player'
    return max(actions, key=lambda a: Q.get((state, a), 0.0))

def play_game(current_eps, current_alpha):
    board = [0] * 9
    
    # Randomize who starts to make training robust
    player = 1 if random.random() > 0.5 else -1
    
    prev_canonical_state = None
    prev_action = None
    
    while True:
        # 1. Get state relative to CURRENT player
        canonical_state = get_canonical_state(board, player)
        actions = legal_moves(board)
        
        # 2. Choose action
        action = choose_action(board, player, current_eps)
        
        # 3. Apply move
        next_board = board[:]
        next_board[action] = player
        result = check_win(next_board)
        
        done = False
        reward = 0
        if result is not None:
            done = True
            if result == 0:
                reward = DRAW_REWARD
            elif result == player: # I won
                reward = WIN_REWARD
            else: # I lost (Impossible in this step, but good safety)
                reward = LOSS_REWARD

        # 4. UPDATE OPPONENT (The one who moved previously)
        # We must look at the board from the OPPONENT'S perspective (-player)
        if prev_canonical_state is not None:
            if done:
                # If I won, opponent gets negative reward
                target = -1 * reward 
            else:
                # Opponent's target is -1 * My Best Value
                # (My best value is derived from MY perspective)
                best_val_my_perspective = max([Q.get((canonical_state, a), 0.0) for a in actions], default=0)
                target = -1 * best_val_my_perspective
            
            old_q = Q.get((prev_canonical_state, prev_action), 0.0)
            Q[(prev_canonical_state, prev_action)] = old_q + current_alpha * ((gamma * target) - old_q)

        # 5. UPDATE SELF (Only on game end)
        if done:
            old_q = Q.get((canonical_state, action), 0.0)
            Q[(canonical_state, action)] = old_q + current_alpha * (reward - old_q)
            break

        # 6. Store state for next turn
        prev_canonical_state = canonical_state
        prev_action = action
        board = next_board
        player *= -1

# --- TRAINING ---
N = 500_000 
print("Training with CANONICAL states...")

for i in range(1, N + 1):
    progress = i / N
    curr_eps = 1.0 - (progress / 0.9) if progress < 0.9 else 0.0
    curr_alpha = max(0.1, 0.5 * (0.999995 ** i))
    
    play_game(curr_eps, curr_alpha)
    
    if i % 100000 == 0:
        print(f"Game {i} | Size={len(Q)}")

# --- SANITY CHECK ---
print("\n--- FINAL TEST ---")
# Board: O at 0,3. X at 1,4. 
# Relative to O (who is -1), we must FLIP the board for lookup.
# O's perspective: My pieces (-1) become 1. Enemy (1) becomes -1.
# Real Board: [-1, 1, 0, -1, 1, 0, 0, 0, 0]
# Canonical:  [ 1,-1, 0,  1,-1, 0, 0, 0, 0] <-- THIS is what we query
test_board_canonical = (1, -1, 0, 1, -1, 0, 0, 0, 0)
test_actions = [2, 5, 6, 7, 8]

print(f"Canonical State (Player Perspective): {test_board_canonical}")

best_move = -1
best_val = -9999

for m in test_actions:
    val = Q.get((test_board_canonical, m), 0.0)
    label = "[WIN]" if m == 6 else ("[BLOCK]" if m == 7 else "")
    print(f"Move {m} {label}: {val:.2f}")
    if val > best_val:
        best_val = val
        best_move = m

if best_move == 6:
    print("SUCCESS: AI correctly identified the win!")
else:
    print("FAIL: Still failing.")

# --- EXPORT ---
Q_json = {}
for (state, action), value in Q.items():
    state_str = str(list(state)).replace(" ", "")
    key = f"{state_str}_{action}"
    Q_json[key] = round(value, 4)

with open("q.json", "w") as f:
    json.dump(Q_json, f)
print("Saved q.json")