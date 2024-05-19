import streamlit as st
import replicate
import os
from transformers import AutoTokenizer
import whisper
from fpdf import FPDF
import numpy as np
import torchaudio
import time
import random

# Set assistant icon to text
icons = {"assistant": "🤖", "user": "🎓"}

# App title and initial upload
st.set_page_config(page_title="AI-Generated University Lecture Notes")
st.title("Welcome to Notebolt!")
st.write("Hi, I'm your virtual assistant here to help you generate structured notes from your lecture audios.")

# Welcome Screen
with st.expander("Learn More About Notebolt"):
    st.write("""
    **Notebolt** helps students take better notes during lectures by providing:
    - Transcription of lecture audio.
    - Highlighting key points and summarizing content.
    - Interactive key points for deeper insights.
    - Downloadable notes in text and PDF formats.
    """)
    #st.image("https://notebolt_image_link_here.jpg", use_column_width=True)

# Upload Lecture
uploaded_file = st.file_uploader("Upload your lecture audio file", type=["wav", "mp3"])

# Replicate Credentials
with st.sidebar:
    st.title('Notebolt: Lecture Notes Generator')
    replicate_api = st.text_input('Enter Replicate API token:', type='password')
    if not (replicate_api.startswith('r8_') and len(replicate_api) == 40):
        st.warning('Please enter a valid Replicate API token.', icon='⚠️')
        st.stop()

    os.environ['REPLICATE_API_TOKEN'] = replicate_api

    st.subheader("Adjust model parameters")
    temperature = st.slider('temperature', min_value=0.01, max_value=5.0, value=0.3, step=0.01)
    top_p = st.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)

    # Display notes in the sidebar
    if 'notes' in st.session_state:
        st.write("### Generated Notes")
        st.write(st.session_state.notes)

# Whisper model for transcription
@st.cache_resource(show_spinner=False)
def get_whisper_model():
    return whisper.load_model("base")

whisper_model = get_whisper_model()

# Function to transcribe audio
def transcribe_audio(file):
    audio, rate = torchaudio.load(file)
    if rate != 16000:
        resampler = torchaudio.transforms.Resample(rate, 16000)
        audio = resampler(audio)
    audio = audio.squeeze().numpy()
    result = whisper_model.transcribe(audio)
    return result["text"]

# Function for generating Snowflake Arctic response
def generate_arctic_response(prompt_str):
    responses = []
    for event in replicate.stream("snowflake/snowflake-arctic-instruct",
                                  input={"prompt": prompt_str,
                                         "temperature": temperature,
                                         "top_p": top_p,
                                         }):
        responses.append(str(event))
    return "".join(responses)

# Clear chat history
def clear_chat_history():
    st.session_state.notes = None

st.sidebar.button('Clear chat history', on_click=clear_chat_history)

# List of fun facts
fun_facts = [
    "Fun Fact: Did you know that the Eiffel Tower can be 15 cm taller during the summer due to the expansion of iron?",
    "Fun Fact: Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still edible.",
    "Fun Fact: Bananas are berries, but strawberries aren't."
]

# Generate and display notes only once per upload
if uploaded_file:
    if "transcription" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.processing = True
        with st.spinner("Processing your lecture..."):
            transcription = transcribe_audio(uploaded_file)
            st.session_state.transcription = transcription
            st.session_state.uploaded_file_name = uploaded_file.name
            
            # Progress bar
            progress_bar = st.progress(0)
            
            # Placeholder for fun facts
            fact_placeholder = st.empty()
            
            # Show progress and fun facts
            for i in range(100):
                time.sleep(0.05)
                progress_bar.progress(i + 1)
                
                # Update the fun fact at intervals
                if i % 33 == 0:
                    random_fact = random.choice(fun_facts)
                    fact_placeholder.info(random_fact)
            
            prompt = f"""user\n{transcription}\nassistant\nGenerate a concise summary of the above lecture notes in the following format:

Title of Lecture

Main Idea: Summarize the main idea of the lecture here.

Key Points: Summarize key points related to the lecture.

Examples:

* Example 1: Provide relevant examples.
* Example 2:
* Example 3:

Additional Points (Professor's Emphasis):

* Include any additional points emphasized by the professor.

Review Questions (to be completed):

* Provide review questions related to the lecture content.

- NOTE: You MUST bolden the headers (like Title of Lecture, Main Idea, Key Points, Examples, Additional Points (Professor's Emphasis), and Review Questions (to be completed)
"""
            response = generate_arctic_response(prompt)
            st.session_state.notes = response  # Store the generated notes
            st.session_state.processing = False
    
    st.write("### Notebolt Summary")
    st.write(st.session_state.notes)

    # Download button for notes in txt format
    st.download_button(
        label="Download Notes as .txt",
        data=st.session_state.notes,
        file_name="lecture_notes.txt",
        mime="text/plain"
    )

    # Create PDF
    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 12)
            self.cell(0, 10, "Lecture Notes", 0, 1, "C")

        def chapter_title(self, title):
            self.set_font("Arial", "B", 12)
            self.cell(0, 10, title, 0, 1, "L")
            self.ln(10)

        def chapter_body(self, body):
            self.set_font("Arial", "", 12)
            self.multi_cell(0, 10, body)
            self.ln()

    pdf = PDF()
    pdf.add_page()
    pdf.chapter_title("Generated Lecture Notes")
    pdf.chapter_body(st.session_state.notes)

    pdf_output = pdf.output(dest="S").encode("latin1")

    # Download button for notes in PDF format
    st.download_button(
        label="Download Notes as PDF",
        data=pdf_output,
        file_name="lecture_notes.pdf",
        mime="application/pdf"
    )

# Chat input for further insights
user_input = st.chat_input("Ask more about the notes...", disabled=('notes' not in st.session_state))
if user_input and 'notes' in st.session_state:
    with st.chat_message("user", avatar=icons["user"]):
        st.write(user_input)

    # Use the previously generated notes for context
    prompt = f"User asks: {user_input}\nNotes Context:\n{st.session_state.notes}\nAssistant:"
    response = generate_arctic_response(prompt)
    with st.chat_message("assistant", avatar=icons["assistant"]):
        st.write_stream(response)
