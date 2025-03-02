from chromadb import PersistentClient
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import project settings
from ragstack.config.settings import settings

print("Testing ChromaDB connection...")

# Connect to local ChromaDB using persistence path
client = PersistentClient(path=os.path.join(project_root, "chromadb-data"))

# List all collections
print("Available collections:")
collections = client.list_collections()
print(collections)

if not collections:
    print("Creating test collection...")
    collection = client.create_collection(name="test_collection")
    
    # Add some test documents
    collection.add(
        documents=["This is a test document", "Another test document"],
        metadatas=[{"source": "test"}, {"source": "test"}],
        ids=["doc1", "doc2"]
    )
    
    # Query the collection
    print("Query results:")
    results = collection.query(
        query_texts=["test document"],
        n_results=2
    )
    print(results)
else:
    # In ChromaDB v0.6.0, list_collections only returns collection names
    print(f"Using existing collection: {collections[0]}")
    collection = client.get_collection(collections[0])
    
    # Count documents
    count = collection.count()
    print(f"Documents in collection: {count}")
    
    if count > 0:
        # Query the collection
        print("Query results:")
        results = collection.query(
            query_texts=["test"],
            n_results=2
        )
        print(results)

