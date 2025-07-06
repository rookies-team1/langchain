from langchain_google_genai import ChatGoogleGenerativeAI

def load_llm():
    # llm 모델
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    return llm