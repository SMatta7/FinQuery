import os
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import streamlit as st
from langchain.llms import OpenAI
from constants import RDS_HOST, RDS_PORT, RDS_USER, RDS_PASSWORD, RDS_DB_NAME, OPENAI_API_KEY

# Editable Settings UI
st.set_page_config(page_title="Conversational Data Analyst Bot", layout="wide")
st.title("Conversational Data Analyst")

with st.sidebar.expander("⚙️ Settings", expanded=True):
    RDS_USER = st.text_input("RDS_USER", os.getenv("RDS_USER", ""))
    RDS_PASSWORD = st.text_input("RDS_PASSWORD", os.getenv("RDS_PASSWORD", ""), type="password")
    RDS_HOST = st.text_input("RDS_HOST", os.getenv("RDS_HOST", "localhost"))
    RDS_PORT = st.text_input("RDS_PORT", os.getenv("RDS_PORT", "3306"))
    RDS_DB_NAME = st.text_input("RDS_DB_NAME", os.getenv("RDS_DB_NAME", ""))
    OPENAI_API_KEY = st.text_input("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""), type="password")

    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

    if st.button("📂 Save and Connect"):
        try:
            db_url = f"mysql+mysqlconnector://{RDS_USER}:{RDS_PASSWORD}@{RDS_HOST}:{RDS_PORT}/{RDS_DB_NAME}"
            engine = create_engine(db_url, pool_pre_ping=True)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            st.session_state.db_engine = engine
            st.success("✅ Connected successfully!")
            st.rerun()
        except Exception as e:
            st.session_state.db_engine = None
            st.error(f"❌ Connection failed: {e}")

# Use saved DB engine if exists
engine = st.session_state.get("db_engine")

# Memory
if 'memory' not in st.session_state:
    st.session_state.memory = []

if 'download_data' not in st.session_state:
    st.session_state.download_data = None

### MODULES / AGENTS ###

# Schema Agent

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

# SQL Generator Agent
def generate_sql(user_text):
    if not engine:
        return "Database not connected."
    from openai import OpenAI as OpenAIClient
    schema_info = get_database_schema()
    history_prompt = "\n\n".join([f"Q: {item['query']}\nSQL: {item['sql']}" for item in st.session_state.memory[-3:]])
    prompt = (
        f"You are a helpful and conversational data analyst agent. Based on the schema and conversation history, generate ONLY a valid MySQL SQL query for the user's request."
        f" Infer table names if needed. Return only the SQL query, with no explanation or prefix.\n\nSchema:\n{schema_info}\n\nConversation History:\n{history_prompt}\n\nUser: {user_text}"
    )
    client = OpenAIClient(api_key=os.environ['OPENAI_API_KEY'])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You generate syntactically correct MySQL queries based on schema and conversation. Do not include any explanation or label like 'SQL:' before the query."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()

# SQL Validator Agent
def validate_sql(query):
    if not engine:
        return False, "Database not connected."
    try:
        with engine.connect() as conn:
            conn.execution_options(autocommit=True)
            conn.execute(text(f"EXPLAIN {query}"))
        return True, None
    except SQLAlchemyError as e:
        return False, str(e)

# SQL Executor Agent
def run_query(query):
    if not engine:
        return None, "Database not connected."
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        return df, None
    except SQLAlchemyError as e:
        return None, str(e)

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
    except Exception:
        return None

# Summary Agent
def generate_summary(df):
    try:
        from openai import OpenAI as OpenAIClient
        prompt = "Please summarize this dataset in plain language:\n" + df.head(20).to_csv(index=False)
        client = OpenAIClient(api_key=os.environ['OPENAI_API_KEY'])
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Summary unavailable."

### Chat Logic ###
user_query = st.chat_input("Ask me something about your database...")
if user_query:
    if not engine:
        st.error("❌ Please connect to the database first.")
    else:
        st.chat_message("user").markdown(user_query)

        with st.spinner("Let me think... 🧠"):
            sql = generate_sql(user_query)
            st.chat_message("assistant").markdown(f"Here's the SQL I came up with:\n```sql\n{sql}\n```")

            valid, error = validate_sql(sql)
            if not valid:
                st.chat_message("assistant").error(f"Hmm, I couldn't validate that SQL: {error}")
            else:
                df, err = run_query(sql)
                if err:
                    st.chat_message("assistant").error(f"Oops, something went wrong: {err}")
                elif df is not None:
                    st.chat_message("assistant").success("Here's what I found:")
                    st.dataframe(df)

                    st.session_state.download_data = df.copy()

                    if df.shape[1] >= 2:
                        fig = generate_plot(df)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

                    summary = generate_summary(df)
                    st.markdown("**📄 Summary:**")
                    st.write(summary)

                    st.session_state.memory.append({"query": user_query, "sql": sql, "summary": summary})

# Sidebar: History + Downloads
st.sidebar.title("🗂️ Conversation History")
if st.session_state.download_data is not None:
    st.sidebar.download_button(
        label="📅 Download Last Result",
        data=st.session_state.download_data.to_csv(index=False),
        file_name="query_result.csv",
        mime="text/csv"
    )

for i, m in enumerate(reversed(st.session_state.memory)):
    st.sidebar.markdown(f"**Q:** {m['query']}")
    st.sidebar.code(m['sql'], language="sql")
    if st.sidebar.button(f"▶️ Run again {i}", key=f"rerun_{i}"):
        st.session_state.user_query = m['query']
        st.rerun()