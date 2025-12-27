import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from textblob import TextBlob
import re
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Summarization function
def summarize_text(text, max_length=50000):
    summarization_pipeline = pipeline("summarization")
    summary = summarization_pipeline(text, max_length=max_length, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Keyword extraction
def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum()]
    keywords = [word for word in words if word not in stop_words and len(word) > 1]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([' '.join(keywords)])
    vocabulary = vectorizer.vocabulary_
    top_keywords = sorted(vocabulary, key=vocabulary.get, reverse=True)[:5]

    return top_keywords

# Topic modeling
def topic_modeling(text):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tf = vectorizer.fit_transform([text])
    lda_model = LatentDirichletAllocation(n_components=5, max_iter=5, learning_method='online', random_state=42)
    lda_model.fit(tf)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        topics.append([feature_names[i] for i in topic.argsort()[:-6:-1]])
    return topics

# Extract YouTube video ID from URL
def extract_video_id(url):
    patterns = [
        r'v=([^&]+)',
        r'youtu.be/([^?]+)',
        r'youtube.com/embed/([^?]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Main app
def main():
    st.title("YouTube Video Summarizer")

    video_url = st.text_input("Enter YouTube Video URL:")
    max_summary_length = st.slider("Max Summary Length:", 1000, 20000, 5000)

    if st.button("Summarize"):
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("Invalid YouTube URL. Please enter a valid URL.")
            return

        try:
            # Get transcript (works with youtube-transcript-api==1.2.3)
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            video_text = ' '.join([line['text'] for line in transcript])

            # Summarize
            summary = summarize_text(video_text, max_length=max_summary_length)

            # Keywords
            keywords = extract_keywords(video_text)

            # Topics
            topics = topic_modeling(video_text)

            # Sentiment
            sentiment = TextBlob(video_text).sentiment

            # Display
            st.subheader("Video Summary:")
            st.write(summary)

            st.subheader("Keywords:")
            st.write(keywords)

            st.subheader("Topics:")
            for idx, topic in enumerate(topics):
                st.write(f"Topic {idx+1}: {', '.join(topic)}")

            st.subheader("Sentiment Analysis:")
            st.write(f"Polarity: {sentiment.polarity}")
            st.write(f"Subjectivity: {sentiment.subjectivity}")

        except TranscriptsDisabled:
            st.error("Transcripts are disabled for this video.")
        except NoTranscriptFound:
            st.error("No transcript found for this video.")
        except Exception as e:
            st.error(f"Transcript not available: {str(e)}")

if __name__ == "__main__":
    main()
