import streamlit as st
import subprocess
import re
from transformers import pipeline
from textblob import TextBlob

def extract_video_id(url):
    match = re.search(r"(v=|youtu.be/)([^&]+)", url)
    return match.group(2) if match else None

def get_captions(video_id):
    cmd = [
        "yt-dlp",
        "--write-auto-sub",
        "--sub-lang", "en",
        "--skip-download",
        f"https://www.youtube.com/watch?v={video_id}"
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    for file in os.listdir():
        if file.endswith(".vtt"):
            with open(file, "r", encoding="utf-8") as f:
                return f.read()
    return None

def summarize(text):
    summarizer = pipeline("summarization")
    return summarizer(text, max_length=250, min_length=60, do_sample=False)[0]["summary_text"]

st.title("YouTube Video Summarizer")

url = st.text_input("Enter YouTube URL")

if st.button("Summarize"):
    video_id = extract_video_id(url)
    if not video_id:
        st.error("Invalid URL")
        st.stop()

    captions = get_captions(video_id)
    if not captions:
        st.error("No captions available for this video")
        st.stop()

    summary = summarize(captions)
    sentiment = TextBlob(captions).sentiment

    st.subheader("Summary")
    st.write(summary)

    st.subheader("Sentiment")
    st.write(sentiment)
