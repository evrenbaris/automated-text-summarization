from transformers import pipeline

# Load the summarization model from Hugging Face
summarizer = pipeline("summarization", model="t5-small")

# Example text for summarization
text = """
Unmanned aerial vehicles (UAVs) used in the defense industry are becoming increasingly advanced.
These vehicles offer significant advantages, such as operational flexibility and reduced human casualties.
However, protecting these technologies from cyberattacks is also critical.
In the future, UAVs will have even more enhanced AI-supported analysis capabilities.
"""

# Perform summarization
summary = summarizer(text, max_length=50, min_length=20, do_sample=False)
print("Summary:", summary[0]['summary_text'])
