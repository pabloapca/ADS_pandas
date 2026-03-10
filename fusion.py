import pandas as pd
import glob
import os

# 1. Setup paths
input_path = './all_covid_data' 
output_file = 'fused_election_data.csv'
all_files = glob.glob(os.path.join(input_path, "*.csv"))

print(f"🚀 Found {len(all_files)} files. Starting Fusion...")

# 2. Optimization: Define only the columns you need for the report
# This prevents your computer from crashing on 917 files
keep_cols = ['created_at', 'tweet', 'likes', 'retweet_count', 'user_followers_count', 'state', 'country']

df_list = []

for i, filename in enumerate(all_files):
    try:
        # lineterminator='\n' handles messy tweet newlines
        # low_memory=False prevents DtypeWarnings
        temp_df = pd.read_csv(filename, lineterminator='\n', usecols=keep_cols, low_memory=False)
        
        # Add a source column (useful for Outlier Investigation later)
        temp_df['source_file'] = os.path.basename(filename)
        
        df_list.append(temp_df)
        
        if i % 100 == 0:
            print(f"✅ Merged {i} files...")
            
    except Exception as e:
        print(f"⚠️ Skipping {filename} due to error: {e}")

# 3. Final Concatenation
print("🔗 Finalizing concatenation...")
full_df = pd.concat(df_list, axis=0, ignore_index=True)

# 4. Filter for US Only (As per your report scope)
full_df = full_df[full_df['country'] == 'United States of America']

# 5. Save for Analysis
full_df.to_csv(output_file, index=False)
print(f"🎉 Success! Fused dataset saved as {output_file}")
print(f"Total Tweets: {len(full_df)}")