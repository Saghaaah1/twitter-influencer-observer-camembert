import json, re
import pandas as pd
import numpy as np
from collections import Counter

rows = []
with open('train.jsonl') as f:
    for line in f:
        rows.append(json.loads(line))

# Extract source app name
def parse_source(s):
    if not s: return 'unknown'
    m = re.search(r'>([^<]+)<', str(s))
    return m.group(1).strip() if m else str(s)[:50]

sources = [(parse_source(r.get('source','')), r['label']) for r in rows]
src_df = pd.DataFrame(sources, columns=['source','label'])

print("Top 20 sources by count and label rate:")
grp = src_df.groupby('source').agg(count=('label','count'), influencer_rate=('label','mean')).sort_values('count', ascending=False)
print(grp.head(20).to_string())

# Check timestamp patterns
ts = [r.get('timestamp_ms') for r in rows[:10]]
print('\nTimestamp samples:', ts[:5])
