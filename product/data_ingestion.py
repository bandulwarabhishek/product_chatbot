

from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from product.data_converter import DataConverter
from product.config import Config
from utils.logger import get_logger

logging = get_logger(__name__)

class DataIngestion:
    def __init__(self):
        self.embedding = HuggingFaceEndpointEmbeddings(model=Config.EMBEDDING_MODEL)

        try:
            logging.info(f"{'='*20}Intialization started.{'='*20}")
            self.vectorstore = AstraDBVectorStore(
            embedding = self.embedding,
            api_endpoint = Config.ASTRA_DB_API_ENDPOINT,
            token = Config.ASTRA_DB_APPLICATION_TOKEN,
            namespace = Config.ASTRA_DB_KEYSPACE,
            collection_name = "product_chatbot_db"
        )
        except Exception as e:
            logging.error(f"Error in intialization: {e} ")
             
#load_existing - if docs already exists

    def ingest(self, load_existing=True):

        try:
            logging.info(f"{'='*20}Data Ingestion started.{'='*20}")
            if load_existing == True:
                 return self.vectorstore
        
            docs = DataConverter("data/flipkart_product_review.csv").converter()

            self.vectorstore.add_documents(docs)

            return self.vectorstore

        except Exception as e:
            logging.error(f"Error in data ingestion: {e}")
    

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.ingest(load_existing=False)
