## Ingest code - you may need to run this the first time
# Load
import os

from langchain.document_loaders import DocugamiLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

if __name__ == "__main__":
    DOCUGAMI_API_KEY = os.environ.get("DOCUGAMI_API_KEY")
    PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX", "langchain-test")

    loader = DocugamiLoader(
        docset_id="4eik4b8kuoup",
        document_ids=["4vevuqrux0yb"],
        min_text_length=32,
        max_text_length=1024 * 8,
        sub_chunk_tables=False,
        xml_mode=True,
        parent_hierarchy_levels=2
    )

    data = loader.load()

    # Add to vectorDB
    vectorstore = Pinecone.from_documents(
        documents=data, embedding=OpenAIEmbeddings(), index_name=PINECONE_INDEX_NAME
    )
    retriever = vectorstore.as_retriever()
