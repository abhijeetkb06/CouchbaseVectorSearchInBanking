import streamlit as st
import json
from datetime import timedelta
from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.exceptions import CouchbaseException

# Initialize the same embedding model you were using
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# Define global variables for Couchbase connection
cluster = None
vector_store = None

def connect_to_couchbase():
    """Establishes connection to the Couchbase cluster and initializes vector store."""
    global cluster, vector_store
    if cluster is None or vector_store is None:
        try:
            # Connection setup following official LangChain docs
            auth = PasswordAuthenticator('admin', 'Password@P1')
            options = ClusterOptions(auth)
            cluster = Cluster('couchbases://cb.puo-rfi1metq3bnn.cloud.couchbase.com', options)
            cluster.wait_until_ready(timedelta(seconds=10))
            
            # Initialize CouchbaseSearchVectorStore
            vector_store = CouchbaseSearchVectorStore(
                cluster=cluster,
                bucket_name='products',
                scope_name='_default',
                collection_name='_default',
                embedding=embeddings,
                index_name='vector_index',  # Your search index name
                text_key='description',     # Field containing text content
                embedding_key='vector'      # Field containing vector embeddings
            )
            st.info("Connected to Couchbase with LangChain.")
        except CouchbaseException as e:
            st.error(f"Failed to connect to Couchbase: {e}")

def insert_products_into_couchbase(products):
    """Inserts products into Couchbase using LangChain CouchbaseSearchVectorStore."""
    global vector_store, cluster
    if not vector_store:
        return
    
    try:
        # Check if data already exists (same logic as your original)
        bucket = cluster.bucket('products')
        if bucket.default_collection().exists(products[0]['productId']).exists:
            return
            
        # Convert products to LangChain Documents
        documents = []
        ids = []
        
        for product in products:
            # Create Document with description as page_content
            doc = Document(
                page_content=product['description'],
                metadata={
                    'productId': product['productId'],
                    'productName': product['productName']
                }
            )
            documents.append(doc)
            ids.append(product['productId'])
        
        # Add documents to vector store (automatically handles vectorization)
        vector_store.add_documents(documents=documents, ids=ids)
        st.success(f"Loaded {len(products)} products into the database.")
        
    except CouchbaseException as e:
        st.error(f"Failed to load product data into Couchbase: {e}")

def perform_product_search(query_text, k=5):
    """Performs similarity search using LangChain CouchbaseSearchVectorStore."""
    global vector_store
    if not vector_store:
        return None
    
    try:
        # Use similarity_search_with_score (replaces your manual vector search)
        results = vector_store.similarity_search_with_score(
            query=query_text,
            k=k,
            fields=["productName", "description"]  # Specify fields to return
        )
        return results
    except CouchbaseException as e:
        st.error(f"Product search failed: {e}")
        return None

def load_sample_data():
    """Load sample transaction data from a JSON file."""
    with open('data/Products.json', 'r') as sample_data:
        products = json.load(sample_data)
    return products

def main():
    st.markdown("""
    <style>
    .title-font {
        font-size: 28px;
        font-weight: bold;
    }
    .powered-font {
        color: red;
        font-size: 20px;
    }
    .product-name {
        color: #6495ED; 
        font-weight: bold;
    }
    .score-color {
        color: #00FFD1; 
    }
    </style>
    <div>
        <span class="title-font">Product Recommendation System</span><br>
        <span class="powered-font">Powered By Couchbase Vector Search & LangChain</span>
    </div>
    """, unsafe_allow_html=True)
    
    connect_to_couchbase()

    # Load and insert products into Couchbase
    products = load_sample_data()
    insert_products_into_couchbase(products)

    user_query = st.text_input("What are you looking for?")
    if user_query:
        results = perform_product_search(user_query)
        if results:
            for doc, score in results:
                # Extract information from Document and metadata
                product_name = doc.metadata.get('productName', 'Unknown Product')
                description = doc.page_content
                
                # Format display with same styling as your original
                product_name_html = f"<span class='product-name'>{product_name}</span>"
                score_html = f"<span class='score-color'>Score: {score:.4f}</span>"
                st.markdown(f"{product_name_html} - {description} - {score_html}", 
                          unsafe_allow_html=True)
        else:
            st.write("No products found matching your query.")

if __name__ == "__main__":
    main()
