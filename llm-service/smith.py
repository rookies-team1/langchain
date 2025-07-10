from langchain_openai import ChatOpenAI
from langsmith import Client
import os
from langsmith import traceable



LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "llm-service-already")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true").lower() == "true"
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

api_key = os.getenv("LANGSMITH_API_KEY")
client = Client(api_key=api_key)



@traceable(run_type="chain", name="Simple_Chain")
def simple_chain(input_data):

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.7
    )

    result = llm.invoke(input_data)
    
    # try:
    #     projects = client.get_projects()
    #     print("API 인증 성공, 프로젝트 목록:", projects)
    # except Exception as e:
    #     print("API 인증 실패:", e)

    # print(os.environ.get("LANGSMITH_API_KEY"))
    print("Simple Chain Result:", result)
    return result

answer = simple_chain("hello groq")
print("Answer:", answer)