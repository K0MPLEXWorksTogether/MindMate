import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    temperature=0,
    groq_api_key= os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-70b-versatile"
)

history = []

def processTextWithLlm(Text, emotionString):
    history_text = "\n".join(history)
    
    prompt_template = PromptTemplate.from_template(
        """
        ### Conversation History:
        {history_text}
        
        ### TEXT From User:
        {Text}
        
        ### Emotion String:
        {emotionString}

        ### INSTRUCTION:
        The above given text is a user's conversation. A professional therapist has identified the following emotions and passed them to you as an emotion string. You need to make a conversation, considering yourself to be a indirect therapist. You must not simply process these emotions passed to you, but rather try to nudge your partner towards the right way to handle the situation described to you. Again, you must respond like you are having a normal conversation with a friend, but just try to nudge your partner towards handling the situation correctly. In the given emotions, consider an emotion only if it's value is greater than or equal to 20%.

        ### ANSWER:
        """
    )
    
    chain = prompt_template | llm
    response = chain.invoke(input={'history_text': history_text, 'Text': Text, 'emotionString': emotionString})

    history.append(f"User: {Text}")
    history.append(f"Therapist: {response.content}")

    return response.content