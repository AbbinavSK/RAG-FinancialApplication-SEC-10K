'''Build a Financial analysis RAG app with Llama-Index to generate insights about SEC 10-K documents.'''

import os
from sec_edgar_downloader import Downloader
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.palm import PaLM
from llama_index.embeddings.google import GooglePaLMEmbedding

# Add the Palm API key into the environment
os.environ['PALM_API_KEY'] = "AIzaSyDfjMWDcmm8DnOse3VeqknwQbih6Yb6uKI"

# Load the documents
documents = SimpleDirectoryReader("/workspaces/RAG-FinancialApplication-SEC-10K/data").load_data()
print("Successfully Loaded", len(documents), "documents !!")

# Create the word to vector index and vector query engine
llm = PaLM()
embed_model = GooglePaLMEmbedding(model="models/embedding-gecko-001", embed_batch_size=1000)
vector_query_engine = VectorStoreIndex.from_documents(documents, embed_model = embed_model, show_progress=True).as_query_engine(llm = llm)
print("Successfully created vector indices !!")

# Create the subquestion query engine to enhance the vector query engine
query_engine_tools = [
        QueryEngineTool(
            query_engine = vector_query_engine,
            metadata = ToolMetadata(
                name = "SEC_10-K_Reports",
                description = f"Provides information about Apple and Pepsico annual financials from 1995 to 2023.",
            ),
        ),
    ]

query_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools = query_engine_tools, 
                                                    llm = llm)

# Process the query and get the response
response = query_engine.query(
    """Your are an accountant and you are going to analyse the financials for Apple and Pepsico 
    using the SEC 10-K documents."""
    )
print(response)
