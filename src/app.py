from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Annotated

from src.templates import prabhupad_system_prompt

class State(TypedDict):
    # Messages are of type (role: str, content: str)
    messages: Annotated[list, "The messages in the conversation"]
    context: Annotated[str, "Retrieved context"]

def format_docs(docs):
    return "\n\n".join(f"Chapter {doc.metadata.get('chapter', 'N/A')}, Verse {doc.metadata.get('verse', 'N/A')}: {doc.page_content}" for doc in docs)

def app(retriever, llm):
    # Define the prompt
    prompt = ChatPromptTemplate([
        ("system", "{system_prompt}"),
        ("system", "Optional Context: {context}"),
        ("placeholder", "{conversation}"),
    ])
    chain = prompt | llm
    # Define the nodes
    def retrieve_context(state: State) -> State:
        question = state["messages"][-1][1]
        docs = retriever.get_relevant_documents(question)
        new_context = format_docs(docs)
        combined_context = f"{state.get('context', '')}\n\n{new_context}".strip()
        return State(messages=state["messages"], context=combined_context)
    def generate_response(state: State) -> State:
        response = chain.invoke({
            "system_prompt": prabhupad_system_prompt,
            "context": state["context"],
            "conversation": state["messages"],
        })
        new_messages = state["messages"] + [("assistant", response.content)]
        return State(messages=new_messages, context=state["context"])
    # Set up the graph
    graph = StateGraph(State)
    graph.add_node("retriever", retrieve_context)
    graph.add_node("generator", generate_response)
    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", END)
    # Compile the graph
    return graph.compile()
