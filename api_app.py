from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from main import get_app_instance
from src.app import State

# Initialize the FastAPI app
api_app = FastAPI()

# Initialize the chatbot instance
app = get_app_instance()

# Define the request and response model for the API
class Message(BaseModel):
    role: str
    content: str
class QueryRequest(BaseModel):
    question: str
    messages: list[Message] = []
class QueryResponse(BaseModel):
    response: str

# Define the API endpoint to query the chatbot
@api_app.post("/query", response_model=QueryResponse)
def query_model(request: QueryRequest):
    """
    Handles a query to the chatbot.

    Args:
        request (QueryRequest): The user's question and optional previous messages.

    Returns:
        QueryResponse: The chatbot's answer.
    """
    try:
        message_tuples = [(message.role, message.content) for message in request.messages]
        state = State(messages=[*message_tuples, ("user", request.question)], context="")
        new_state = app.invoke(state)
        return QueryResponse(response=new_state["messages"][-1][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the query: {str(e)}")

@api_app.get("/")
def read_root():
    """
    Returns a simple message indicating that the API is running.
    """
    return {"message": "API is running"}