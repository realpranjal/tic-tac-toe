import pickle
import json

with open("q.pkl", "rb") as f:
    Q = pickle.load(f)

Q_json = {}
for (state, action), value in Q.items():
    key = f"{state}_{action}"
    Q_json[key] = value

with open("q.json", "w") as f:
    json.dump(Q_json, f)

print(f"Converted {len(Q)} Q-values to q.json")
