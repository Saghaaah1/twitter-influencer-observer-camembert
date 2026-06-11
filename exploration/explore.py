import json
import pandas as pd
from pandas import json_normalize

# Load and explore data
with open('train.jsonl') as f:
    d = json.loads(f.readline())

print('Top-level keys:', list(d.keys()))
print('Label:', d.get('label'))
print('challenge_id:', d.get('challenge_id'))

# Check nested keys
for k, v in d.items():
    if isinstance(v, dict):
        print(f'  {k} (dict) keys: {list(v.keys())[:10]}')
    elif isinstance(v, list):
        print(f'  {k} (list) len={len(v)}')
    else:
        print(f'  {k}: {str(v)[:80]}')

print('\n--- Loading full dataset ---')
train_data = pd.read_json('train.jsonl', lines=True)
train_data = json_normalize(train_data.to_dict(orient='records'))
print('Shape:', train_data.shape)
print('Columns sample:', list(train_data.columns)[:30])
print('Label distribution:\n', train_data['label'].value_counts())

# Show non-null counts for interesting columns
numeric_cols = train_data.select_dtypes(include='number').columns.tolist()
print('\nNumeric columns:', numeric_cols[:20])
