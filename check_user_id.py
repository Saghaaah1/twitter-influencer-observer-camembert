import json, pandas as pd, numpy as np
from pandas import json_normalize

with open('train.jsonl') as f:
    d = json.loads(f.readline())

print("User dict keys:", list(d['user'].keys()))
print()
print("Has 'id'?", 'id' in d['user'])
print("Has 'id_str'?", 'id_str' in d['user'])
print("Has 'screen_name'?", 'screen_name' in d['user'])
print()

# Check what user identifier we DO have
user = d['user']
for k, v in user.items():
    if k in ['id', 'id_str', 'screen_name', 'name']:
        print(f"user.{k} = {v}")

# Check if challenge_id can serve as user identifier
print()
# Load several rows and see if challenge_id pattern reveals user
rows = []
with open('train.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 20: break
        rows.append(json.loads(line))

print("challenge_id samples:", [r['challenge_id'] for r in rows[:5]])
print("id_str samples:", [r.get('id_str', '?') for r in rows[:5]])

# Check user screen_name stability
users = [r['user'] for r in rows]
print("\nscreen_name samples:", [u.get('screen_name', 'MISSING') for u in users[:10]])
print("label samples:", [r['label'] for r in rows[:10]])
