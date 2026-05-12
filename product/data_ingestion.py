from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from product.data_converter import DataConverter
from product.config import Config
from utils.logger import get_logger

logging = get_logger(__name__)

class DataIngestion:
    def __init__(self):
        # Validate credentials before initialization
        self._validate_credentials()
        
        self.embedding = HuggingFaceEndpointEmbeddings(model=Config.EMBEDDING_MODEL)
        self.vectorstore = None

        try:
            logging.info(f"{'='*20}Initialization started.{'='*20}")
            logging.info(f"Connecting to AstraDB with endpoint: {Config.ASTRA_DB_API_ENDPOINT}")
            self.vectorstore = AstraDBVectorStore(
                embedding=self.embedding,
                api_endpoint=Config.ASTRA_DB_API_ENDPOINT,
                token=Config.ASTRA_DB_APPLICATION_TOKEN,
                namespace=Config.ASTRA_DB_KEYSPACE,
                collection_name="product_chatbot_db"
            )
            logging.info("✅ Successfully connected to AstraDB")
        except Exception as e:
            logging.error(f"Error in initialization: {e}")
            raise
    
    def _validate_credentials(self):
        """Validate that all required credentials are present"""
        required_vars = {
            "ASTRA_DB_API_ENDPOINT": Config.ASTRA_DB_API_ENDPOINT,
            "ASTRA_DB_APPLICATION_TOKEN": Config.ASTRA_DB_APPLICATION_TOKEN,
            "ASTRA_DB_KEYSPACE": Config.ASTRA_DB_KEYSPACE,
        }
        
        for var_name, var_value in required_vars.items():
            if not var_value:
                error_msg = f"Missing environment variable: {var_name}. Please set it in your .env file"
                logging.error(error_msg)
                raise ValueError(error_msg)
            
            # Check for common issues
            if var_name == "ASTRA_DB_APPLICATION_TOKEN" and not var_value.startswith("AstraCS:"):
                logging.warning(f"⚠️  Warning: {var_name} should start with 'AstraCS:'. Check if it's correctly copied.")
            
            if var_name == "ASTRA_DB_API_ENDPOINT" and not var_value.startswith("https://"):
                logging.warning(f"⚠️  Warning: {var_name} should start with 'https://'. Check if it's correctly copied.")
             
    #load_existing - if docs already exists

    def ingest(self, load_existing=True):

        try:
            if self.vectorstore is None:
                raise RuntimeError("Vector store initialization failed. Cannot proceed with ingestion.")
                
            logging.info(f"{'='*20}Data Ingestion started.{'='*20}")
            if load_existing:
                logging.info("Loading existing vector store...")
                return self.vectorstore
        
            logging.info("Converting data from CSV...")
            docs = DataConverter("data/flipkart_product_review.csv").converter()
            
            logging.info(f"Adding {len(docs)} documents to vector store...")
            self.vectorstore.add_documents(docs)
            logging.info("✅ Data ingestion completed successfully")

            return self.vectorstore

        except Exception as e:
            logging.error(f"❌ Error in data ingestion: {e}")
            raise
    

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.ingest(load_existing=False)
