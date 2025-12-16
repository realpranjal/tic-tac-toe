import random
import json

# load trained Q-table
with open("q.json", "r") as f:
    Q_json = json.load(f)

# Convert back to tuple keys for Python
Q = {}
for key, value in Q_json.items():
    state_str, action = key.rsplit('_', 1)
    state = tuple(eval(state_str))
    Q[(state, int(action))] = value

eps = 0.0

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

def legal_moves(board):
    return [i for i in range(9) if board[i] == 0]

# --- NEW: Helper to flip board perspective ---
def get_canonical_state(board, player):
    # If I am Player 1, board is normal.
    # If I am Player -1, flip all pieces (-1 becomes 1, 1 becomes -1).
    if player == 1:
        return tuple(board)
    else:
        return tuple([-x for x in board])

def choose_action(board, player, actions):
    # 1. Flip the board so AI sees itself as '1'
    state = get_canonical_state(board, player)
    
    # 2. Query Q-table with the flipped state
    # Breaking ties randomly is important so it doesn't just pick the first index (0)
    random.shuffle(actions)
    return max(actions, key=lambda a: Q.get((state, a), 0))

def print_board(b):
    s = {1:"X", -1:"O", 0:"."}
    print()
    for i in range(0,9,3):
        print(" ".join(s[b[i+j]] for j in range(3)))

# game loop
board = [0]*9

player = random.choice([1, -1])
   
if player == -1:
    print("AI goes first! (O)")
else:
    print("You go first! (X)")

while True:
    print_board(board)

    if player == 1: # Human (X)
        try:
            move = int(input("Your move (1-9): ")) - 1
            if move not in legal_moves(board):
                print("Invalid move")
                continue
            board[move] = 1
        except ValueError:
            print("Please enter a number.")
            continue
    else: # AI (O)
        # Pass the full board and the current player ID (-1)
        move = choose_action(board, player, legal_moves(board))
        board[move] = -1
        print(f"AI plays {move + 1}")

    result = check_win(board)
    if result is not None:
        print_board(board)
        if result == 1: print("You win!")
        elif result == -1: print("AI wins!")
        else: print("Draw!")
        break

    player *= -1