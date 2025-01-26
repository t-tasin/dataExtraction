import pandas as pd
import numpy as np
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage

import os
from llama_index.core import Settings
import logging
from dotenv import load_dotenv
from io import StringIO
from typing import Optional
from datetime import datetime
import time
   
load_dotenv()

azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_api_type = os.getenv("OPENAI_API_TYPE")
azure_openai_chat_deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")


llm = AzureOpenAI(
    model='gpt-4',
    deployment_name=azure_openai_chat_deployment_name,
    api_key=azure_openai_api_key,
    azure_endpoint=azure_openai_endpoint,
    api_version=azure_openai_api_version,
    temperature=0
)

path = "../datasets/synth_data.ods"
df = pd.read_excel(path)
df_copy = df.copy()

def filter_df(columns:list, values:list):
    """
    Filters the global DataFrame 'df' based on the provided columns and corresponding values.
    
    This function iterates over the given columns and values, and filters the DataFrame 'df'
    such that only the rows where the column values match the specified values are retained.
    
    Parameters:
    columns (list): A list of column names to filter by.
    values (list): A list of values corresponding to each column name in 'columns' to filter for. The 
    values inside of this list cannot be lists or dictionaries
    
    Returns:
    pd.DataFrame: The filtered DataFrame with rows matching the specified column values.

    """
    global df
    for column, value in zip(columns, values):
        df = df.loc[df[column]==value]
    return df

def sum_df(column: str):
    """
    Calculate the sum of a specified column in the global DataFrame 'df'.

    Parameters:
    column (str): The column name whose values are to be summed.
    
    Returns:
    Numeric value
    """
    
    global df
    if column in df.columns:
        return df[column].sum()
    else:
        raise ValueError(f"Column '{column} does not exist in the DataFrame.") 

def groupby_function(groupby_column: str, agg_column: str, agg_func: str):
    """
    Group by a specified column and aggregate another column with a specified function.
    
    Parameters:
    groupby_column (str): The column to group by.
    agg_column (str): The column to aggregate.
    agg_func (str): The aggregation function to use (e.g., 'sum', 'mean', 'count').

    Returns:
    pd.DataFrame: The resulting DataFrame after groupby and aggregation.
    """
    global df 
    grouped_df = df.groupby(groupby_column)[agg_column].agg(agg_func).reset_index()
    df = grouped_df
    return grouped_df

def max_df(column: str) -> pd.DataFrame:
    """
    Find the maximum value in a specified column of the global DataFrame 'df'.

    Parameters:
    column (str): The column name to find the maximum value for.

    Returns:
    Numeric Value
    """
    global df
    return df[column].max()

#Finds the smallest value in the dataset for a specified column
def min_df(column: str):
    """
    Find the minimum value in a specified column of the global DataFrame 'df'.

    Parameters:
    column (str): The column name to find the minimum value for.

    Returns:
    Numeric Value
    """
    global df
    return df[column].min()

def select_columns(columns):
    """
    Select specified columns from the global DataFrame 'df'.

    Parameters:
    columns (list): A list of column names to select from the DataFrame.

    Returns:
    pd.DataFrame: A DataFrame containing only the specified columns.

    Raises:
    ValueError: If any of the specified columns do not exist in the DataFrame.
    """
    global df
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
    df = df[columns]
    return df[columns]

#Used to sort by ascending/descending values
def sort(column: str, ascending: bool):
    """
    This tool is useful for sorting a dataframe by a user specified column. 
    
    Ascending should be true if you want to sort by ascending values 
    and false if you want to sort by descending values.
    
    Parameters:
    column (str): The column name to sort by.
    ascending (bool): If True, sort in ascending order, otherwise sort in descending order.
    """
    global df
    df = df.sort_values(by=column,ascending=ascending)
    return df.sort_values(by=column,ascending=ascending)

def average_df(column: str):
    """
    Calculate the average (mean) of a specified column in the global DataFrame 'df'.

    Parameters:
    column (str): The column name whose values are to be averaged.

    Returns:
    Numeric Value
    """
    global df
    return df[column].mean()

def select_a_number_of_rows(num_rows: int):
    """
    Select a specified top number of rows from the dataset. 

    Parameters:
    num_rows (int): The number of rows to select.

    Returns:
    pd.DataFrame: The selected rows from the DataFrame.
    """
    global df
    df = df.head(num_rows)
    return df.head(num_rows)

def percent(values: dict):
    """
    Calculate the percentage of each value in a specified column relative to the total sum of that column.

    Parameters:
    column (str): The name of the column to calculate percentages for. If the column name does not exist in the original 
    dataframe do not use the column parameter 
    values (dict) 

    Returns:
    pd.DataFrame: A DataFrame with the original values and their corresponding percentages.
    """
    global df
    total_sum = sum(values.values())
    percentages = {key: (value / total_sum) * 100 for key, value in values.items()}
    return percentages
        

def anomolies_df(column: str, threshold: float = 3.0):
    """
    Identifies anomolies by calculating the Z-Score.

    Parameters:
    column (str): The column name to calculate Z-scores for.
    threshold (float): The Z-score threshold to use for identifying anomalies. Default is 3.0.
    
    Returns:
    A dataset containing anomlies
    """
    global df
    
    column_mean = df[column].mean()
    column_std = df[column].std()
    
    z_scores = (df[column] - column_mean) / column_std
    anomalies = df[np.abs(z_scores) > threshold]
    return anomalies
    
def reset_df(): 
    """
    Used to reset the dataframe back to its original schema and values
    
    Returns: 
    pd.DataFrame: The reset Dataframe 'df'
    """
    global df, df_copy
    df = df_copy
    return df
    
#Others - contains, type conversion,
#, .dt.month (allows you to extract the month from the datetime column)

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

filter_tool = FunctionTool.from_defaults(fn=filter_df)
sum_tool = FunctionTool.from_defaults(fn=sum_df)
group_tool = FunctionTool.from_defaults(fn=groupby_function)
findMax_tool = FunctionTool.from_defaults(fn=max_df)
select_columns_tool = FunctionTool.from_defaults(fn = select_columns)
sort_tool = FunctionTool.from_defaults(fn=sort)
findMin_tool = FunctionTool.from_defaults(fn=min_df)
percent_tool = FunctionTool.from_defaults(fn=percent)
anomolies_tool = FunctionTool.from_defaults(fn=anomolies_df)
select_rows_tool = FunctionTool.from_defaults(fn=select_a_number_of_rows)
average_tool = FunctionTool.from_defaults(fn=average_df)
reset_tool = FunctionTool.from_defaults(fn=reset_df)

 
#Agent
# agent = ReActAgent.from_tools(
#     tools=[filter_tool, sum_tool, group_tool, findMax_tool, select_columns_tool, sort_tool,findMin_tool, percent_tool, 
#            anomolies_tool, select_rows_tool,average_tool
#            ],
#     verbose=True,
#     max_iterations=50,
#     llm=llm,
#     context = ReAct_Context
# )


prompts = ["What is the total energy consumption for Haryana in Janurary 2023?",
"Which site has the highest consumption of electricity consumption in December 2021",
"What percentage of data is estimated versus measured in the United States?",
"What is the total water usage for all sites in India for May 2021?",
"How does the energy mix (proportion of electric, gas, water) vary by site in Canada",
"Are there any anomalies in the data that suggest Ontario_Canada_164 consumption is outside of expected ranges?",
"Which sites have shown the greatest improvement in reducing consumption or emissions over time?",
"What is the total water usage for all sites in India for May 2021?",
"Can we identify any seasonal patterns in energy or water usage across sites?",
]
# prompts = ["Which site has the largest decrease in emissions between 2021 and 2023?"]

print(f"Current Date and Time: {datetime.now()}")
#prompts = ["What percentage of data is estimated versus measured in the United States?"]
# prompt = """Find all the values of data_stream Electric Power for state Haryana and give me total value."""
# prompt = """Find the Elevation of all values for Ohio"""
start_time = time.time()
i = 0
# while i < 3: 
#     for prompt in prompts:
#         df = df.copy()
#         agent = ReActAgent.from_tools(
#             tools=[filter_tool, sum_tool, group_tool, findMax_tool, select_columns_tool, sort_tool,findMin_tool, percent_tool, 
#            anomolies_tool, select_rows_tool,average_tool, reset_tool],
#             verbose=True,
#             max_iterations=50, 
#             llm=llm,
#             context=ReAct_Context
#         )
#         print(f"Prompt: {prompt}\n")
#         response = agent.chat(
#             prompt
#         )
#         response = str(response)
#         print('\n')
#         print(response)
#         print('\n')
#         print('--------------------------------------------------------------------------------------------------------------------------------\n--------------------------------------------------------------------------------------------------------------------------------\n')
#     i = i + 1
# end_time = time.time()
# print(end_time - start_time)

class PromptTransformer:
    def __init__(self, llm):
        self.llm = llm

    def transform(self, user_input: str) -> str:
        prompt = f"Rewrite the following user query in a very specific and detailed way to interact with the padnas dataframe {df}: {user_input}"
        messages = [
            ChatMessage(role="system", content="You are a senior prompt engineer"),
            ChatMessage(role="user", content= prompt)
        ]
        response = self.llm.chat(messages)
        return response

llm_transformer = AzureOpenAI(
    model='gpt-4',
    deployment_name=azure_openai_chat_deployment_name,
    api_key=azure_openai_api_key,
    azure_endpoint=azure_openai_endpoint,
    api_version=azure_openai_api_version,
    temperature=0
)

prompt_transformer = PromptTransformer(llm_transformer)

userQuery = " Which site has the largest decrease in emissions between 2021 and 2023?"
transformedQuery = prompt_transformer.transform(userQuery)
transformed_query = str(transformedQuery)
print(transformed_query)


agent = ReActAgent.from_tools(
    tools=[filter_tool, sum_tool, group_tool, findMax_tool, select_columns_tool, sort_tool,findMin_tool, percent_tool, 
    anomolies_tool, select_rows_tool,average_tool, reset_tool],
    verbose=True,
    max_iterations=50, 
    llm=llm,
    context=ReAct_Context
)
# print(f"Prompt: {prompt}\n")
response = agent.chat(
    transformed_query
)
response = str(response)
print('\n')
print(response)
print('\n')
print('--------------------------------------------------------------------------------------------------------------------------------\n--------------------------------------------------------------------------------------------------------------------------------\n')
end_time = time.time()
print(end_time - start_time)