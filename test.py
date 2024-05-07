'''Build a Financial analysis RAG app with Llama-Index to generate insights about SEC 10-K documents.'''

import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.palm import PaLM
from llama_index.embeddings.google import GooglePaLMEmbedding

# Add the Palm API key into the environment
os.environ['PALM_API_KEY'] = "AIzaSyDfjMWDcmm8DnOse3VeqknwQbih6Yb6uKI"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ='AppData\Roaming\gcloud\application_default_credentials.json'

# Load the documents
documents = SimpleDirectoryReader("data").load_data()
print("Successfully Loaded", len(documents), "documents !!")

# Create the word to vector index and vector query engine
llm = PaLM()
embed_model = GooglePaLMEmbedding(model="models/embedding-gecko-001", embed_batch_size=10000)
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
    using the SEC 10-K documents. Answer the following questions in great detail:

    1. How is the revenue growth for Apple and Pepsico CAGR from 1995 to 2023?
    2. What is the average gross profit margin for Apple and Pepsico since 1995?
    3. What is the expenses growth vs profit growth for both companies? Is there a greater disparity
        with time?
    4. Similar to the previous question, how is distribution and marketing cost growth for Apple and Pepsico
        in relation to revenue growth? Is there a greater disparity with time?
    5. What is the average adjusted earnings per share for Apple and Pepsico since 1995? Which company has performed better?
    6. What is the CAGR for shareholders equity for both companies?
    7. Has the Adjusted book value grown over time? How much has it grown?
    8. What is assets/liabilities ratio since 1995?
    
    For each of the above question, answer in detail and produce visualisation using matplotlib or plotly 
    to illustrate the answers."""
    )
print(response)
