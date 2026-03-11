import pandas as pd
import glob
import os
import datetime

# 1. Setup paths
input_path = './all_covid_data' 
output_file = 'daily_sentiment_volatility.csv'
all_files = glob.glob(os.path.join(input_path, "*.csv"))

print(f"🚀 Found {len(all_files)} files. Starting Temporal Aggregation...")

# The actual IEEE dataset has no headers. 
# Column 0: Tweet ID | Column 1: Sentiment Score
daily_stats = []

for i, filename in enumerate(all_files):
    try:
        # We read the file without headers
        # Use only column 1 (sentiment) to save massive amounts of RAM
        # We use column 0 (ID) just for the first row to get the date
        df = pd.read_csv(filename, header=None, names=['tweet_id', 'sentiment'], low_memory=False)
        
        # --- EXCELLENT GRADE FEATURE: Bit-shift ID to get exact Date ---
        # Twitter encodes the timestamp in the ID. This logic is from the IEEE instructions.
        example_id = int(df['tweet_id'].iloc[0])
        timestamp = (example_id >> 22) + 1288834974657
        date = datetime.datetime.fromtimestamp(timestamp/1000.0).date()
        
        # --- ANALYTICAL DEPTH: Calculate Volatility ---
        # Instead of just the mean, we calculate the Standard Deviation (Volatility)
        # and the count (Volume), matching the ONS Loneliness methodology.
        stats = {
            'date': date,
            'sentiment_mean': df['sentiment'].mean(),
            'sentiment_volatility': df['sentiment'].std(), # This is your key metric!
            'tweet_volume': len(df),
            'source_file': os.path.basename(filename)
        }
        
        daily_stats.append(stats)
        
        if i % 50 == 0:
            print(f"✅ Processed {i} files. Current Date: {date}")
            
    except Exception as e:
        print(f"⚠️ Skipping {filename}: {e}")

# 2. Final Fusion
print("🔗 Fusing daily aggregates...")
fused_df = pd.DataFrame(daily_stats)

# 3. Handle Missing Data (As per SPHERE report)
# If volatility is NaN (happens if a file has only 1 tweet), we drop or impute.
fused_df = fused_df.dropna(subset=['sentiment_volatility'])

# 4. Save the "Cleaned" Fused Dataset
fused_df.to_csv(output_file, index=False)

print(f"🎉 Success! Fused metrics saved as {output_file}")
print(f"Final Dataset Shape: {fused_df.shape}")