import os
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.tools import tool
from constants import RDS_HOST, RDS_PORT, RDS_USER, RDS_PASSWORD, RDS_DB_NAME, OPENAI_API_KEY

# Editable Settings UI
st.set_page_config(page_title="Conversational Data Analyst Bot", layout="wide")
st.title("🧠 Conversational Data Analyst")

with st.sidebar.expander("⚙️ Settings", expanded=True):
    RDS_USER = st.text_input("RDS_USER", os.getenv("RDS_USER", ""))
    RDS_PASSWORD = st.text_input("RDS_PASSWORD", os.getenv("RDS_PASSWORD", ""), type="password")
    RDS_HOST = st.text_input("RDS_HOST", os.getenv("RDS_HOST", "localhost"))
    RDS_PORT = st.text_input("RDS_PORT", os.getenv("RDS_PORT", "3306"))
    RDS_DB_NAME = st.text_input("RDS_DB_NAME", os.getenv("RDS_DB_NAME", ""))
    #OPENAI_API_KEY = st.text_input("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""), type="password")

    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

    if st.button("📂 Save and Connect"):
        try:
            db_url = f"mysql+mysqlconnector://{RDS_USER}:{RDS_PASSWORD}@{RDS_HOST}:{RDS_PORT}/{RDS_DB_NAME}"
            engine = create_engine(db_url, pool_pre_ping=True)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            st.session_state.db_engine = engine
            st.success("✅ Connected successfully!")
        except Exception as e:
            st.session_state.db_engine = None
            st.error(f"❌ Connection failed: {e}")

# Use saved DB engine if exists
engine = st.session_state.get("db_engine")

# Memory for context
memory = ConversationBufferMemory(memory_key="chat_history")

def build_schema_graph():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT TABLE_NAME, COLUMN_NAME, COLUMN_KEY, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE TABLE_SCHEMA = :schema
            """), {"schema": RDS_DB_NAME})

            relations = []
            for row in result:
                if row[3] and row[4]:
                    relations.append(f"{row[0]}.{row[1]} → {row[3]}.{row[4]}")

            return "\n".join(relations)
    except Exception as e:
        return f"Could not build schema graph: {e}"

# === GRAPH REASONING ===
def get_database_schema():
    if not engine:
        return "Database not connected."
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT TABLE_NAME, COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = :schema
                ORDER BY TABLE_NAME, ORDINAL_POSITION
            """), {"schema": RDS_DB_NAME})

            schema_dict = {}
            for table_name, column_name in result:
                schema_dict.setdefault(table_name, []).append(column_name)

            schema_text = ""
            for table, columns in schema_dict.items():
                schema_text += f"Table: {table}\nColumns: {', '.join(columns)}\n\n"
            return schema_text

    except Exception as e:
        return f"Could not retrieve schema information: {e}"


@tool
def generate_sql(query: str) -> str:
    """Generates SQL based on natural language query."""
    schema = get_database_schema()
    relationships = build_schema_graph()
    prompt = (
        f"You are a data analyst AI. Use the following schema and foreign key relationships to generate a valid MySQL SQL query."
        f"\n\nSchema:\n{schema}\n\nRelationships:\n{relationships}\n\nUser Query: {query}"
        f"\n\nMake sure to use appropriate JOINs when data spans multiple tables. Return only the SQL query, no explanation."
    )
    llm = OpenAI(temperature=0, openai_api_key=os.environ['OPENAI_API_KEY'])
    return llm(prompt)

@tool
def validate_sql(query: str) -> str:
    """Validates SQL query."""
    try:
        with engine.connect() as conn:
            conn.execute(text(f"EXPLAIN {query}"))
        return "✅ SQL is valid"
    except Exception as e:
        return f"❌ Invalid SQL: {e}"

@tool
def run_sql(query: str) -> str:
    """Runs SQL and stores result in memory."""
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        st.session_state.result_df = df
        return df.head(5).to_markdown()
    except Exception as e:
        return f"❌ Query failed: {e}"

def summarize_result() -> str:
    """Summarizes or visualizes query result."""
    try:
        df = st.session_state.get("result_df")
        if df is None:
            return "No results to summarize."
        
        print("********************")
        print(df.select_dtypes(include='number').shape[1])
        if df.select_dtypes(include='number').shape[1] >= 1:
            fig = generate_plot(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                #return "📊 Visualized the data using your chosen chart type."

        # If not visual, fall back to summary
        prompt = "Summarize the following data:\n" + df.head(20).to_csv(index=False)
        llm = OpenAI(temperature=0, openai_api_key=os.environ['OPENAI_API_KEY'])
        return llm(prompt)
    except Exception as e:
        return f"Failed to summarize or visualize: {e}"

# Plot Agent
def generate_plot(df):
    try:
        cols = list(df.columns)
        chart_type = st.selectbox("Choose chart type", ["Bar", "Line", "Pie"], key="chart_type")

        if len(cols) >= 2:
            if chart_type == "Bar":
                fig = px.bar(df, x=cols[0], y=cols[1])
            elif chart_type == "Line":
                fig = px.line(df, x=cols[0], y=cols[1])
            elif chart_type == "Pie":
                fig = px.pie(df, names=cols[0], values=cols[1])
        elif len(cols) == 1:
            fig = px.histogram(df, x=cols[0])
        else:
            return None

        return fig
    except Exception as e:
        st.error(f"Plotting failed: {e}")
        return None


# === AGENT ===
tools = [generate_sql, validate_sql, run_sql]
ag = initialize_agent(tools, OpenAI(temperature=0, openai_api_key=os.environ['OPENAI_API_KEY']), agent_type="zero-shot-react-description", memory=memory, verbose=True)

# === CHAT UI ===
user_query = st.chat_input("Ask your database question...")
if user_query:
    st.chat_message("user").markdown(user_query)
    with st.spinner("Thinking..."):
        result = ag.run(user_query)
        st.chat_message("assistant").markdown(result)

        # Optional summarization logic
        #if "summarize" in user_query.lower() or "summary" in user_query.lower():
        summary = summarize_result()
        st.markdown("#### 📊 Summary:")
        st.markdown(summary)
        if st.session_state.get("result_df") is not None:
            st.dataframe(st.session_state["result_df"])
