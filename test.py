import pandas as pd

# Files downloaded from Kaggle
biden_file = 'hashtag_joebiden.csv'
trump_file = 'hashtag_donaldtrump.csv'

import pandas as pd

# 1. Load with line terminator
biden = pd.read_csv('hashtag_joebiden.csv', lineterminator='\n')
trump = pd.read_csv('hashtag_donaldtrump.csv', lineterminator='\n')

# 2. Add labels and merge
biden['candidate'] = 'biden'
trump['candidate'] = 'trump'
df = pd.concat([biden, trump]).drop_duplicates(subset='tweet_id')

# 3. Fix dates
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

# 4. Filter for US only (optional)
df = df[df['country'] == 'United States of America']

try:
    # Load only 1,000 rows to test
    # lineterminator='\n' is crucial for this dataset because of tweet formatting
    df_biden = pd.read_csv(biden_file, lineterminator='\n', nrows=1000)
    df_trump = pd.read_csv(trump_file, lineterminator='\n', nrows=1000)

    print("✅ Success! Files loaded correctly.")
    print(f"Biden dataset sample: {df_biden.shape}")
    print(f"Trump dataset sample: {df_trump.shape}")
    
    # Peek at the data
    print("\nFirst 5 Biden tweets:")
    print(df_biden['tweet'].head())

except FileNotFoundError:
    print("Error: CSV files not found. Did the 'kaggle download' finish?")
except Exception as e:
    print(f"An error occurred: {e}")