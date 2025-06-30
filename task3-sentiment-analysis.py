import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Step 1: Load both files
df_train = pd.read_csv("task3-sentiment-patterns-twitter/twitter_training.csv", header=None)
df_valid = pd.read_csv("task3-sentiment-patterns-twitter/twitter_validation.csv", header=None)

# Step 2: Combine them
df = pd.concat([df_train, df_valid], ignore_index=True)

# Step 3: Rename columns (based on dataset structure)
df.columns = ['tweet_id', 'entity', 'sentiment', 'content']

# Step 4: Drop missing content/sentiment
df = df.dropna(subset=['content', 'sentiment'])

# Step 5: Lowercase sentiment values
df['sentiment'] = df['sentiment'].str.lower()

# Step 6: Plot sentiment distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='sentiment', order=df['sentiment'].value_counts().index, palette='coolwarm')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Tweet Count")
plt.tight_layout()
plt.show()

# Step 7: Generate WordClouds per sentiment
def generate_wordcloud(sentiment_label):
    text = " ".join(df[df['sentiment'] == sentiment_label]['content'].astype(str))
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud - {sentiment_label.capitalize()}")
    plt.show()

# Generate word clouds
for sentiment in df['sentiment'].unique():
    generate_wordcloud(sentiment)

# Step 8: Save for Power BI
df.to_csv("twitter_sentiment_results.csv", index=False)
print("✅ Cleaned + combined data saved as 'twitter_sentiment_results.csv'")