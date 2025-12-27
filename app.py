import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from transformers import pipeline
from textblob import TextBlob
import re
import nltk

nltk.download("punkt")

def extract_video_id(url):
    patterns = [
        r"v=([^&]+)",
        r"youtu.be/([^?]+)",
        r"youtube.com/embed/([^?]+)"
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None

def summarize_text(text):
    summarizer = pipeline("summarization")
    result = summarizer(text, max_length=300, min_length=60, do_sample=False)
    return result[0]["summary_text"]

def main():
    st.title("YouTube Video Summarizer")

    url = st.text_input("Enter YouTube URL")

    if st.button("Summarize"):
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Invalid YouTube URL")
            return

        try:
            # ✅ ONLY METHOD THAT WORKS ON STREAMLIT CLOUD
            transcripts, errors = YouTubeTranscriptApi.get_transcripts(
                [video_id],
                languages=["en"]
            )

            if video_id not in transcripts:
                st.error("Transcript not available")
                return

            video_text = " ".join(
                [item["text"] for item in transcripts[video_id]]
            )

            summary = summarize_text(video_text)
            sentiment = TextBlob(video_text).sentiment

            st.subheader("Summary")
            st.write(summary)

            st.subheader("Sentiment")
            st.write(f"Polarity: {sentiment.polarity}")
            st.write(f"Subjectivity: {sentiment.subjectivity}")

        except TranscriptsDisabled:
            st.error("Transcripts are disabled for this video")
        except NoTranscriptFound:
            st.error("No transcript found")
        except Exception as e:
            st.error(str(e))

if __name__ == "__main__":
    main()
