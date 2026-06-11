import json, pandas as pd, numpy as np
from pandas import json_normalize

train = pd.read_json('train.jsonl', lines=True)
train = json_normalize(train.to_dict(orient='records'))
test = pd.read_json('kaggle_test.jsonl', lines=True)
test = json_normalize(test.to_dict(orient='records'))

# Find the right user ID columns
user_id_cols = [c for c in train.columns if c.startswith('user.') and 'id' in c.lower()]
print('User ID cols:', user_id_cols)

# Use user.id or user.id_str equivalent
uid_col = None
for c in ['user.id_str', 'user.id']:
    if c in train.columns:
        uid_col = c
        break

if uid_col is None:
    # Flatten and try user dict
    print('Columns with user:', [c for c in train.columns if c.startswith('user.')][:10])
else:
    train_uids = set(train[uid_col].astype(str))
    test_uids = set(test[uid_col].astype(str))
    overlap = train_uids & test_uids
    print(f'uid_col: {uid_col}')
    print(f'Train unique users: {len(train_uids)}')
    print(f'Test unique users:  {len(test_uids)}')
    print(f'Overlap: {len(overlap)} ({len(overlap)/len(test_uids)*100:.1f}% of test users)')

    # For overlapping users: are their test tweets' labels consistent with training?
    # (we don't have test labels, but check train label consistency)
    train_label = train.groupby(uid_col)['label'].agg(['mean', 'std', 'count'])
    print('\nTrain label consistency (per user):')
    print(f"  Users with all same label (std=0): {(train_label['std']==0).sum()}")
    print(f"  Users with mixed labels: {(train_label['std']>0).sum()}")
    print(f"  Mean label std per user: {train_label['std'].mean():.3f}")
