import streamlit as st
import torch
import scipy.io.wavfile
from io import BytesIO
from transformers import pipeline

# Carreguem el model
synthesiser = st.cache_resource(pipeline)(
    "text-to-audio",
    "facebook/musicgen-small"
)

# Interfície Streamlit
st.title("Music Generator")
st.write("Introdueix una descripció i genera música.")

text = st.text_input(
    "Text",
    placeholder="Describe the music you want to generate..."
)

if st.button("Generate music"):
    if text.strip():
        music = synthesiser(
            text,
            forward_params={"do_sample": True}
        )

        audio_buffer = BytesIO()
        scipy.io.wavfile.write(
            audio_buffer,
            rate=music["sampling_rate"],
            data=music["audio"].squeeze()
        )

        st.audio(audio_buffer.getvalue(), format="audio/wav")
    else:
        st.warning("Introdueix una descripció.")
