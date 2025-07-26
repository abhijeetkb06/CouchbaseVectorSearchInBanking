from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.exceptions import CouchbaseException
import streamlit as st
from sentence_transformers import SentenceTransformer
from datetime import timedelta
import couchbase.search as search
from couchbase.vector_search import VectorQuery, VectorSearch
import json
import numpy as np

# Initialize the NLP model globally to avoid reloading it on each function call
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define  variables for Couchbase connection
cluster = None
bucket = None

def connect_to_couchbase():
    """Establishes connection to the Couchbase cluster and bucket."""
    global cluster, bucket
    if cluster is None or bucket is None:
        try:
            cluster = Cluster('couchbases://cb.puo-rfi1metq3bnn.cloud.couchbase.com',
                              ClusterOptions(PasswordAuthenticator('admin', 'Password@P1')))
            cluster.wait_until_ready(timedelta(seconds=10))
            bucket = cluster.bucket('products')
            st.info("Connected to Couchbase.")
        except CouchbaseException as e:
            st.error(f"Failed to connect to Couchbase: {e}")

def vectorize_description(description):
    """Vectorizes product descriptions."""
    return model.encode(description).tolist()

def insert_products_into_couchbase(products):
    """Inserts products into Couchbase with vector embeddings."""
    global bucket
    if not bucket:
        return
    try:
        # Check if data has already been loaded to avoid redundancy by checking the first product's existence
        if bucket.default_collection().exists(products[0]['productId']).exists:
            # st.info("Sample data already loaded. Skipping re-insertion.")
            return
        for product in products:
            key = product['productId']
            product['vector'] = vectorize_description(product['description'])
            bucket.default_collection().upsert(key, product)
        st.success(f"Loaded {len(products)} products into the database.")
    except CouchbaseException as e:
        st.error(f"Failed to load product data into Couchbase: {e}")

def perform_product_search(query_vector):
    """Performs a vector search in Couchbase for products matching the query."""
    global bucket
    if not bucket:
        return
    search_index = 'vector_index'  # Update with your vector search index name
    try:
        search_req = search.SearchRequest.create(search.MatchNoneQuery()).with_vector_search(
            VectorSearch.from_vector_query(
                VectorQuery('vector', query_vector, num_candidates=5)
            )
        )
        result = bucket.default_scope().search(search_index, search_req, search.SearchOptions(limit=5, fields=["productName", "description", "score"]))
        return result
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
        <span class="powered-font">Powered By Couchbase Vector Search</span>
    </div>
    """, unsafe_allow_html=True)
    connect_to_couchbase()

    # Optionally: Load and insert products into Couchbase (comment out if already done)
    products = load_sample_data()
    insert_products_into_couchbase(products)

    user_query = st.text_input("What are you looking for?")
    if user_query:
        query_vector = vectorize_description(user_query)
        results = perform_product_search(query_vector)
        if results and results.rows():
            for row in results.rows():
                # Use markdown for product name to apply green color and score to apply light blue color
                product_name_html = f"<span class='product-name'>{row.fields.get('productName')}</span>"
                score_html = f"<span class='score-color'>Score: {row.score}</span>"
                st.markdown(f"{product_name_html} - {row.fields.get('description')} - {score_html}", unsafe_allow_html=True)
        else:
            st.write("No products found matching your query.")

if __name__ == "__main__":
    main()


