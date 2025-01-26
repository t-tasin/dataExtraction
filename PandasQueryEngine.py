# Testing Files: "Testing_with_prompting","Testing_without_prompting","ReAct_w_PandasQuery_Tests","PandasQuery_Tests"

# Pandas Parser
# Try to reinitate the LLM before hand
# Try to say that they can only do their tasks by using tools
import sys
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
import numpy as np
from llama_index.llms.azure_openai import AzureOpenAI
import os
import logging
from dotenv import load_dotenv
from llama_index.core import PromptTemplate
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from datetime import datetime
from llama_index.experimental.query_engine.pandas import (
    PandasInstructionParser,)
import time
load_dotenv()

# Setting up the API Information
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_api_type = os.getenv("OPENAI_API_TYPE")
azure_openai_chat_deployment_name = os.getenv(
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

df = pd.read_excel('../datasets/synth_data.ods')

llm = AzureOpenAI(
    model='gpt-4',
    deployment_name=azure_openai_chat_deployment_name,
    api_key=azure_openai_api_key,
    azure_endpoint=azure_openai_endpoint,
    api_version=azure_openai_api_version,
    temperature=0
)

# I slightly changed the prompting so that 'Actual' is assumed unless specified
new_prompt = PromptTemplate(
    """\
You are working with a pandas dataframe in Python.
The name of the dataframe is `df`.
This is the result of `print(df.head())`:
{df_str}

Follow these instructions:
Initiate your query with df[df['type'] == 'Actual'] unless the user specifies otherwise.  
{instruction_str}
Query: {query_str}

Expression: """
)

query_engine = PandasQueryEngine(
    df=df, verbose=True, llm=llm, synthesize_response=True)
query_engine.update_prompts({"pandas_prompt": new_prompt})


# prompts= [
#     "What is the total energy consumption for Haryana in Janurary 2023?",
#     "What is the average monthly energy consumption across all sites?",
#     "Which site has the highest consumption of natural gas, electricity, water, or emissions in December 2021",
#     "How does the energy consumption of Western Australia_Australia_200 compare to other sites in the same state?",
#     "What are the trends in energy consumption in 2023 for Wyoming_United States_239",
#     "How do emissions data correlate with energy consumption across all sites?",
#     "What percentage of data is estimated versus measured across all sites?",
#     "What is the total water usage for all sites in India for May 2021?",
#     "Can we identify any seasonal patterns in energy or water usage across sites?",
#     "How does the energy mix (proportion of electric, gas, water) vary by site in Canada",
#     "Are there any anomalies in the data that suggest Ontario_Canada_164 consumption is outside of expected ranges?",
#     "Which sites have shown the greatest improvement in reducing consumption or emissions over time?"
# ]


# prompts = [
# "What are the top 5 highest sites in 2021 for consumption of electricity?"
# ]

# for prompt in prompts:
#      print(f"Prompt: {prompt}\n")
#      response = query_engine.query(
#          prompt
#      )
#      response = str(response)
#      print('\n')
#      print(response)
#      print('--------------------------------------------------------------------------------------------------------------------------------\n--------------------------------------------------------------------------------------------------------------------------------\n')

pandas_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="pandas_tool",
    description="""This is a querying tool for pandas that should be able to access and manipulate data 
    from datasets and databases, this tool can also be used to do basic mathematical operations to datasets"""
)

ReAct_Context = """
Do not attempt to use specific input (column name, row names, types) until after trying to use the pandas tool to see the column names and rows 
(Make sure to change this information accordingly if needed).
ALWAYS use your tools for prompts. Do not answer without using your tools unless you absolutely have to in order to answer the question.
ALWAYS use your tools to do calculations. 

Dataset Description and Pandas Query Instructions:

This dataset contains information on environmental and energy metrics from various sites. It comprises 9 columns:

An index column (Unnamed) serving as the row identifier.
Columns: site_name, state, country, service_month, data_stream, value, unit, type.
Data Types:

site_name, state, country, data_stream, and type are all text/string values.
service_month is in date format.
The value column consists of numerical values.
The data_stream column contains one of four possible values: Electric power, Natural Gas, Water, or Emissions.

The type column contains one of two possible values: Actual (default) or Forecasted.

All site names end with a numerical identifier. Here is an example of what a site name would look like: Haryana_India_51. 
If a value of a location does not contain this numerical identifer it is a state or a country, not a site name. 


Instructions for Pandas Query:

Use Pandas to load and manipulate the dataset.
Identify column names and inspect the first few rows to understand the data structure.
Assume default querying for Actual data (Type column) unless instructed otherwise. 
DO NOT query for forecasted data unless the user says "forecasted","estimated", or a word with an identical meaning; if the user 
does not say this then you need to filter by Actual (Type column.)
unless the user prompt specifically says to do so.
Sample potential tasks:
Analyzing trends over time.
Comparing values across sites or states.
Aggregating data based on specific criteria.
Please adhere to these instructions while querying the dataset.
Whenever the word "energy" is used you should assume it means electric power and natural gas in the data_stream column. 

Tool Specific Instructions: 
You must insert whether the "ascending" parameter is true or false for the sort tool
You MUST use the reset dataframe tool before you filter from the same column more than once.

Example of sample data from the dataset is below: 

site_name,state,country,service_month,data_stream,value,unit,type
Haryana_India_51,Haryana,India,01/01/2021,Electric Power,12618.88,kWh,Actual
Haryana_India_51,Haryana,India,01/01/2021,Natural Gas,15382.05,MMBtu,Actual
Haryana_India_51,Haryana,India,01/01/2021,Water,15071.40,Gallons,Forecasted

"""

prompts = [
    "What is the total energy consumption for Haryana in Janurary 2023?",
    "Which site has the highest consumption of electricity consumption in December 2021",
    "What percentage of data is estimated versus measured in the United States?",
    "What is the total water usage for all sites in India for May 2021?"
]
print(f"Current Date and Time: {datetime.now()}")
start_time = time.time()
i = 0
while i < 3:
    for prompt in prompts:
        llm = AzureOpenAI(
            model='gpt-4',
            deployment_name=azure_openai_chat_deployment_name,
            api_key=azure_openai_api_key,
            azure_endpoint=azure_openai_endpoint,
            api_version=azure_openai_api_version,
            temperature=0
        )
        agent = ReActAgent.from_tools(
            tools=[pandas_tool],
            verbose=True,
            max_iterations=20,
            llm=llm,
            context=ReAct_Context
        )
        print(f"Prompt: {prompt}\n")
        response = agent.chat(
            prompt
        )
        response = str(response)
        print('\n')
        print(response)
        print('\n')
        print('--------------------------------------------------------------------------------------------------------------------------------\n--------------------------------------------------------------------------------------------------------------------------------\n')
    i = i + 1
end_time = time.time()
print(end_time-start_time)
