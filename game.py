import pickle
import random

# load trained Q-table
with open("q.pkl", "rb") as f:
    Q = pickle.load(f)

eps = 0.0   # NO exploration while playing

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

def choose_action(state, actions):
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
    print("AI goes first!")
else:
    print("You go first!")

while True:
    print_board(board)

    if player == 1:
        move = int(input("Your move (1-9): ")) - 1
        if move not in legal_moves(board):
            print("Invalid move")
            continue
        board[move] = 1
    else:
        state = tuple(board)
        move = choose_action(state, legal_moves(board))
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
