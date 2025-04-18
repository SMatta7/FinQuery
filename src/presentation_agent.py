from langchain_openai import ChatOpenAI
from constants import OPENAI_API_KEY

# Initialize LLM
def initialize_presentation_llm():
    """Set up OpenAI LLM for formatting outputs."""
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",  # Cost-effective model
            api_key=OPENAI_API_KEY,
            temperature=0.2,  # Slightly creative for natural summaries
        )
        print("Presentation LLM initialized!")
        return llm
    except Exception as e:
        print(f"Failed to initialize Presentation LLM: {e}")
        return None

# Process Outputs
def format_output(query, sql_result=None, viz_result=None):
    """Format SQL and/or Visualization outputs into a readable summary."""
    llm = initialize_presentation_llm()
    if not llm:
        return "Error initializing presentation agent."

    try:
        prompt = f"""
        Given the query: '{query}'
        SQL result: {sql_result if sql_result else 'None'}
        Visualization code: {viz_result if viz_result else 'None'}
        Generate a concise, readable summary for a non-technical user. Include natural language explanations
        and, if visualization code is provided, a brief description of what it shows.
        """
        response = llm.invoke(prompt).content
        return f"Summary for '{query}':\n{response}"
    except Exception as e:
        return f"Error formatting output for '{query}': {e}"  

# Test the Presentation Agent
if __name__ == "__main__":
    test_query = "What were last month’s top products?"
    test_sql = "[('Product A', 500), ('Product B', 300)]"
    test_viz = "```python\nimport plotly.express as px\npx.bar(x=['Product A', 'Product B'], y=[500, 300])\n```"
    result = format_output(test_query, test_sql, test_viz)
    print(result)