from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/shriram/Work/Generative AI/LangChain Models/.env")

model = ChatGoogleGenerativeAI(model='gemini-3-flash-preview', )

result: AIMessage = model.invoke(['What is the capital of India'])

print(result.content)