import random

Q = {}
alpha, gamma, eps = 0.3, 0.6, 1.0

def choose_action(state, actions):
  if random.random() < eps:
    return random.choice(actions)
  return max(actions, key=lambda a: Q.get((state, a), 0))

def update(state, action, reward, next_state, next_actions):
  best_next = max([Q.get((next_state, a), 0) for a in next_actions], default = 0)
  Q[(state, action)] = Q.get((state, action), 0) + alpha * (reward + gamma * best_next - Q.get((state, action), 0))

                                                                                               
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

def play_game():
    board = empty_board()
    history = []  # store (state, action, player) tuples

    player = 1
    while True:
        state = encode(board)
        actions = legal_moves(board)
        move = choose_action(state, actions)
        
        history.append((state, move, player))
        board = play(board, move, player)
        result = check_win(board)

        if result is not None:
            # Update all moves in history
            for state, action, p in history:
                reward = result * p  # reward from that player's perspective
                Q[(state, action)] = Q.get((state, action), 0) + alpha * (
                    reward - Q.get((state, action), 0))
            break

        player *= -1


N = 1_000_000
min_eps = 0.01

for i in range(1, N + 1):
    play_game()

    eps = max(min_eps, 1.0 * (0.99995 ** i))

    if i % 5000 == 0:
        print(
            f"Training: {i}/{N} | "
            f"eps={eps:.4f} | "
            f"Q size={len(Q)}"
        )



import json

Q_json = {}
for (state, action), value in Q.items():
    state_str = str(list(state))
    key = f"{state_str}_{action}"
    Q_json[key] = value

with open("q.json", "w") as f:
    json.dump(Q_json, f)

print(f"Saved {len(Q_json)} Q-values to q.json")
