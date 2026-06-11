import json, pandas as pd, numpy as np
from pandas import json_normalize

train = pd.read_json('train.jsonl', lines=True)
train = json_normalize(train.to_dict(orient='records'))
test = pd.read_json('kaggle_test.jsonl', lines=True)
test = json_normalize(test.to_dict(orient='records'))

uid_col = [c for c in train.columns if 'id_str' in c and 'user' in c][0]
print('uid col:', uid_col)

train_uids = set(train[uid_col].astype(str))
test_uids = set(test[uid_col].astype(str))
overlap = train_uids & test_uids

print(f'Train unique users: {len(train_uids)}')
print(f'Test unique users:  {len(test_uids)}')
print(f'Overlap: {len(overlap)} ({len(overlap)/len(test_uids)*100:.1f}% of test)')

# Check label consistency within training users
train_df = train[[uid_col, 'label']].copy()
user_labels = train_df.groupby(uid_col)['label'].nunique()
consistent = (user_labels == 1).sum()
inconsistent = (user_labels > 1).sum()
print(f'\nWithin-user label consistency:')
print(f'  Consistent (same label for all tweets): {consistent} ({consistent/len(user_labels)*100:.1f}%)')
print(f'  Inconsistent (mixed labels): {inconsistent} ({inconsistent/len(user_labels)*100:.1f}%)')

# Show train label distribution per user
user_label = train_df.groupby(uid_col)['label'].mean()
print(f'\nUser-level label mean distribution:')
print(user_label.describe())
