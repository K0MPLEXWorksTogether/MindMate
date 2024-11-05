from response import processTextWithLlm
from emotions import predictEmotion
import streamlit as st

def main():
    st.title("MindMate: An AI Therapist")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    userInput = st.chat_input("Talk to MindMate")

    if userInput:
        with st.chat_message("user"):
            st.markdown(userInput)

        emotionsPercentage = predictEmotion(userInput)
        emotionString = ", ".join([f"{emotion}: {percentage}%" for emotion, percentage in emotionsPercentage.items()])
        response = processTextWithLlm(userInput, emotionString=emotionString)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({
            "role": "user",
            "content": userInput,
        })

        st.session_state.messages.append({
            "role": "bot",
            "content": response,
        })

if __name__ == "__main__":
    main()