from langchain.chat_models import init_chat_model 
#initalization chat model 
from langchain.agents import create_agent
#creates an agent graph that calls tools in a loop until a stopping condition is met
from langchain.agents.middleware  import SummarizationMiddleware
#summarizes conversation history when token limit is reached
#it monitors message token counts and automaticaaly summarizes older messages
from langchain.checkpoint.memory import InMemorySaver  #for debugging and testing purpose
#this checkpoint saver stores checkpoints in memory using a default dictionary.
from langchain.tools import tool
from langchain_core.tools import retriever
from product.config import Config
from utils.logger import get_logger

logging = get_logger(__name__)


def build_retriever_tool(retriever):

    @tool

    def product_retriever_tool(query: str) -> str:
        """
        Retrieves relevant product information based on the query.
        """
    
        try:
            logging.info(f"Retrieving information based on: {query}")
            docs = retriever.invoke(query)
            if not docs:
                logging.info("No relevant information found.")
                return f"Sorry, I couldn't find any relevant information about {query}"
        
            return "\n\n".join(doc.page_content for doc in docs)
    
        except Exception as e:
            logging.error(f"Error in retriever tool: {e}")
            return "Sorry, there was an error retrieving the information. Please try again later."

    return product_retriever_tool


class RAGAgentBuilder:
    def __init__(self, vectorstore):
        self.model = init_chat_model(model = Config.RAG_MODEL)

    def build_agent(self):

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        #converting vectorstore into retriever by using a retriever method
        retriever_tool = build_retriever_tool(retriever)
        #converting retriever into a tool

        agent = create_agent(
            model = self.model,
            tools= [retriever_tool], #tool is nothing but is a retriever which is retrieving information from vectorstore.
            system_prompt="""
You are an e-commerce product assistant bot answering product-related queries 
based on review and titles
And to find the answer always use the product_retriever_tool to retrieve relevant information from the vectorstore

If you don't know the answer, answer politely that i don't know the answer please contact our customer support care team for further assistance.
""",
            checkpointer = InMemorySaver(), #for maintaining chat history
            middleware = [
                SummarizationMiddleware(
                    model = self.model,
                    trigger=("messages", 10),
                    keep=("messages",4),
                )
            ],     
        )

        return agent