import pandas as pd
import numpy as np
import re
import matplotlib
# Force 'Agg' backend to avoid Tkinter/GUI errors on Linux/WSL
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# 1. Setup & Downloads
print("📥 Initializing NLP tools...")
nltk.download('vader_lexicon', quiet=True)
sid = SentimentIntensityAnalyzer()
sns.set_theme(style="whitegrid")

def clean_tweet(text):
    """Cleans URLs, Mentions, and special characters from tweets."""
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+', '', text)           # Remove URLs
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)    # Remove Mentions
    text = re.sub(r'#', '', text)                 # Remove Hashtag symbol
    text = re.sub(r'RT[\s]+', '', text)           # Remove Retweet tags
    text = text.replace('\n', ' ').strip()
    return text

def get_sentiment(text):
    """Calculates VADER compound score."""
    return sid.polarity_scores(text)['compound']

# 2. Loading Data
# Note: For testing, you can add 'nrows=10000' to pd.read_csv
print("🚀 Loading datasets (this may take a minute due to file size)...")
try:
    biden = pd.read_csv('hashtag_joebiden.csv', lineterminator='\n')
    trump = pd.read_csv('hashtag_donaldtrump.csv', lineterminator='\n')
except FileNotFoundError:
    print("❌ Error: CSV files not found. Ensure 'hashtag_joebiden.csv' and 'hashtag_donaldtrump.csv' are in this folder.")
    exit()

# 3. Label and Merge
biden['candidate'] = 'Biden'
trump['candidate'] = 'Trump'
df = pd.concat([biden, trump]).drop_duplicates(subset='tweet_id')

# 4. Pre-processing
print("🧹 Filtering for US-based tweets and cleaning text...")
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
df = df[df['country'] == 'United States of America'].copy()
df = df.dropna(subset=['tweet'])

# 5. Sentiment Analysis
print("🧠 Analyzing sentiment scores...")
df['cleaned_tweet'] = df['tweet'].apply(clean_tweet)
df['sentiment_score'] = df['cleaned_tweet'].apply(get_sentiment)

# Categorize the scores
df['sentiment_type'] = df['sentiment_score'].apply(
    lambda c: 'Positive' if c > 0.05 else ('Negative' if c < -0.05 else 'Neutral')
)

# 6. Basic Statistics Output
print("\n--- Sentiment Summary Statistics ---")
print(df.groupby('candidate')['sentiment_score'].describe())

# 7. Visualization 1: Summary Bar and Count Plots
print("📊 Generating Summary Plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Average Sentiment Score (Fixed palette warning)
sns.barplot(
    data=df, 
    x='candidate', 
    y='sentiment_score', 
    hue='candidate', 
    palette=['#0015BC', '#E91D2E'], 
    ax=ax1,
    legend=False
)
ax1.set_title('Average Sentiment Score (Higher = More Positive)')
ax1.set_ylabel('Compound Score (-1 to 1)')

# Sentiment Category Count
sns.countplot(
    data=df, 
    x='sentiment_type', 
    hue='candidate', 
    palette=['#0015BC', '#E91D2E'], 
    ax=ax2
)
ax2.set_title('Volume of Tweets by Sentiment Category')

plt.tight_layout()
plt.savefig('sentiment_summary.png')
print("✅ Saved: sentiment_summary.png")

# 8. Visualization 2: Sentiment Over Time
print("📈 Generating Time-Series Plot...")
df['date'] = df['created_at'].dt.date
time_df = df.groupby(['date', 'candidate'])['sentiment_score'].mean().reset_index()

plt.figure(figsize=(14, 6))
sns.lineplot(
    data=time_df, 
    x='date', 
    y='sentiment_score', 
    hue='candidate', 
    palette=['#0015BC', '#E91D2E']
)
plt.title('Daily Sentiment Trend (October - November 2020)')
plt.xticks(rotation=45)
plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('sentiment_trend.png')
print("✅ Saved: sentiment_trend.png")

print("\n🎉 Analysis Complete! Check your project folder for the .png result files.")