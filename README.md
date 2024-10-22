# SEC 10-K LLM Application

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This LLM application uses Retrieval Augmented Generation (RAG) capabilities of [LlamaIndex](https://github.com/jerryjliu/llama_index) to generate insights about Apple and Pepsico's SEC 10-K documents.

The SEC 10-K documents are in the form of text files generated by SEC-EDGAR-Downloader module in Python. The data was taken from 1995 to 2023, generating 29 text files per ticker. The text files are then fed into the LlamaIndex framework to process and generate inferences based on the queries asked.

## Inferences generated

- Yearly revenue and profit growth CAGR since 1995
- Gross profit margin since 1995
- Expenses growth vs profit growth since 1995
- Distribution and marketing cost vs revenue generated since 1995
- Average adjusted earnings per share
- CAGR for shareholders equity
- Growth of adjusted book value
- Assets/Liabilities ratio since 1995

## Tech Stack ⚒️

- Frontend
    - [Streamlit](https://streamlit.io/)<br/>
        Using Streamlit, we can create a web application that showcases the insights and visualisations generated by our LLM.
        It is easy to build and provides great user experience.
        

- Backend
    - [Google PaLM](https://ai.google.dev/palm_docs/palm)<br/>
        Using Google's PaLM model, we can convert text into embeddings while also generating insights and visualisations from the queries. 
        It is free to use and is robust to queries and chat type applications.
    
    - [LlamaIndex](https://www.llamaindex.ai/)<br/>
        Using LlamaIndex, we can create a framework for our RAG application to ensure smooth and efficient execution.
        It is easy to build and easier to read through the code.

## Conclusions 📝

This project was created as part of an Internship coding assignment at [GaTech FinTech Lab](https://fintech.gatech.edu/#/). It served as an introduction to creating financial RAG applications using LlamaIndex as backend and Streamlit as frontend. 

Yet to complete user interface using Streamlit. (Will be completed in the near future)