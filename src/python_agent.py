# src/python_agent.py

from langchain_experimental.tools import PythonREPLTool
from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain import hub
from sqlalchemy import create_engine, text
import pandas as pd
from langchain.agents import AgentExecutor 
from constants import OPENAI_API_KEY, LLM_MODEL_NAME

from sqlalchemy import create_engine
from constants import RDS_HOST, RDS_PORT, RDS_USER, RDS_PASSWORD, RDS_DB_NAME
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
import re 
#from python_agent import initialize_python_agent

# def initialize_python_agent(engine):
#     instructions = '''Generate SQL queries from natural language requests, fetch data, and create Plotly visualizations.'''
#     prompt = hub.pull('langchain-ai/openai-functions-template').partial(instructions=instructions)
#     llm = ChatOpenAI(model=LLM_MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0)
#     tools = [PythonREPLTool()]
#     agent = create_openai_functions_agent(llm, tools, prompt)

#     def agent_executor(query):
#         response = agent.invoke({'input': query})['output']
#         sql_query = response.split('```sql')[1].split('```')[0].strip()
#         with engine.connect() as connection:
#             df = pd.read_sql(sql=text(sql_query), con=connection)
#         if df.empty:
#             return 'Insufficient data for visualization.'
#         plotly_query = f'{query}. Data: {df.head().to_markdown()} Generate Plotly code.'
#         plotly_response = agent.invoke({'input': plotly_query})['output']
#         plotly_code = plotly_response.split('```python')[1].split('```')[0].strip()
#         return plotly_code
#     return agent_executor


def initialize_python_agent(engine):
    instructions = '''Generate SQL queries from natural language requests, fetch data, and create Plotly visualizations.'''
    prompt = hub.pull('langchain-ai/openai-functions-template').partial(instructions=instructions)
    llm = ChatOpenAI(model=LLM_MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0)
    tools = [
        QuerySQLDatabaseTool(db=SQLDatabase(engine)),  
        PythonREPLTool()
    ]
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Initialize AgentExecutor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True  # Add this line
    )

    def agent_executor_func(query):
        # Step 1: Generate/execute SQL
        sql_response = agent_executor.invoke({"input": f"{query} - generate AND execute SQL"})
        sql_result = sql_response["output"]
        
        # Step 2: Handle SQL errors
        if "error" in sql_result.lower():
            return f"SQL Error: {sql_result}"
            
        # Step 3: Generate visualization
        plotly_response = agent_executor.invoke({
            "input": f"Create Plotly code for: {query}\nData: {sql_result}"
        })
        return extract_python_code(plotly_response["output"])
    
    return agent_executor_func


def extract_python_code(output: str) -> str:
    """Extract Python code block"""
    match = re.search(r'``````', output, re.DOTALL)
    return match.group(1).strip() if match else output

# Main execution
if __name__ == "__main__":
    DATABASE_URL = f'mysql+mysqlconnector://{RDS_USER}:{RDS_PASSWORD}@{RDS_HOST}:{RDS_PORT}/{RDS_DB_NAME}'
    engine = create_engine(DATABASE_URL)

    agent = initialize_python_agent(engine)
    #query = 'pick top 5 merchant_id'
    query = "What were the top 5 items by sales last month?"
    visualization_code = agent(query)
    print(visualization_code)