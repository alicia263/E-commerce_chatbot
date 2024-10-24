import json
import logging
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import warnings
from tqdm import TqdmWarning

# Suppress tqdm warnings
warnings.filterwarnings('ignore', category=TqdmWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataIngestion:
    def __init__(self, es_url='http://localhost:9200', model_name='multi-qa-MiniLM-L6-cos-v1'):
        self.es_client = Elasticsearch([es_url])
        self.model = SentenceTransformer(model_name)
        
    def load_product_data(self, file_path='../data/products_data.json'):
        """Load product data from JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"Could not find file at {file_path}")
            return []
        except json.JSONDecodeError:
            logging.error("Invalid JSON file")
            return []

    def create_text_representation(self, product):
        """Create a combined text representation for encoding."""
        return f"{product['productName']} {product['category']} {product['productDescription']}"

    def encode_product_vectors(self, product):
        """Encode product information into vectors."""
        try:
            return {
                'productName_vector': self.model.encode(product['productName']).tolist(),
                'description_vector': self.model.encode(product['productDescription']).tolist(),
                'combined_vector': self.model.encode(self.create_text_representation(product)).tolist()
            }
        except Exception as e:
            logging.error(f"Error encoding vectors: {str(e)}")
            return None

    def get_elasticsearch_settings(self):
        """Define Elasticsearch index settings and mappings."""
        return {
            'settings': {
                'number_of_shards': 1,
                'number_of_replicas': 0
            },
            'mappings': {
                'properties': {
                    'id': {'type': 'keyword'},
                    'productName': {'type': 'text'},
                    'productDescription': {'type': 'text'},
                    'price': {'type': 'float'},
                    'category': {'type': 'keyword'},
                    'availableColours': {'type': 'keyword'},
                    'sizes': {'type': 'keyword'},
                    'discount': {'type': 'float'},
                    'productName_vector': {
                        'type': 'dense_vector',
                        'dims': 384,
                        'index': True,
                        'similarity': 'cosine'
                    },
                    'description_vector': {
                        'type': 'dense_vector',
                        'dims': 384,
                        'index': True,
                        'similarity': 'cosine'
                    },
                    'combined_vector': {
                        'type': 'dense_vector',
                        'dims': 384,
                        'index': True,
                        'similarity': 'cosine'
                    }
                }
            }
        }

    def setup_elasticsearch_index(self, index_name='ecommerce-products'):
        """Set up Elasticsearch index with proper settings."""
        try:
            if self.es_client.indices.exists(index=index_name):
                self.es_client.indices.delete(index=index_name)
            
            self.es_client.indices.create(
                index=index_name,
                body=self.get_elasticsearch_settings()
            )
            return True
        except Exception as e:
            logging.error(f"Error setting up Elasticsearch index: {str(e)}")
            return False

    def index_products(self, products, index_name='ecommerce-products'):
        """Index products with their vector representations."""
        try:
            successful_indexes = 0
            for product in tqdm(products, desc="Indexing products"):
                vectors = self.encode_product_vectors(product)
                if not vectors:
                    continue
                
                doc = {
                    'id': product['id'],
                    'productName': product['productName'],
                    'productDescription': product['productDescription'],
                    'price': product['price'],
                    'category': product['category'],
                    'availableColours': product['availableColours'],
                    'sizes': product['sizes'],
                    'discount': product['discount'],
                    **vectors
                }
                
                response = self.es_client.index(
                    index=index_name,
                    id=product['id'],
                    body=doc
                )
                if response['result'] == 'created':
                    successful_indexes += 1
            
            logging.info(f"Successfully indexed {successful_indexes} products")
            return successful_indexes
        except Exception as e:
            logging.error(f"Error during indexing: {str(e)}")
            return 0

    def ingest_data(self, file_path, index_name='ecommerce-products'):
        """Main method to handle complete ingestion process."""
        products = self.load_product_data(file_path)
        if not products:
            return False
        
        if not self.setup_elasticsearch_index(index_name):
            return False
            
        return self.index_products(products, index_name) > 0

if __name__ == "__main__":
    ingestion = DataIngestion()
    success = ingestion.ingest_data('data/products_data.json')
    if success:
        logging.info("Data ingestion completed successfully")
    else:
        logging.error("Data ingestion failed")