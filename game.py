import random

Q = {}
alpha, gamma, ep = 0.1, 0.9, 0.5

def choose_action(state, actions):
  if random.random() < eps:
    return random.choices(actions)
  return max(actions, key=lambda a: Q.get((state, a), 0)

