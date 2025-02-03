import streamlit as st
from phi.agent import Agent
from phi.agent import Agent, RunResponse
from phi.utils.pprint import pprint_run_response
from phi.model.google import Gemini
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
Gemini.api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key = Gemini.api_key)
llm_model = genai.GenerativeModel(model_name = 'gemini-1.5-flash', system_instruction = '''You are a stock Analyst
                                                                                        which gives the on point
                                                                                        summary of the analysis provided
                                                                    in the form of a table or text. You need to extract
                                                                    key points which captures the essense beautifully.
                                                                    Please give the response in points with suitable
                                                                    sections (if any)''')

# Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for accurate and up-to-date financial information",
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[DuckDuckGo()],
    instructions=[
        "1. Perform targeted searches to find the most relevant and recent financial information.",
        "2. Always verify the credibility of sources before including them in the response.",
        "3. Summarize the information concisely and include direct links to the sources.",
        "4. If no relevant information is found, state this clearly and suggest alternative search terms.",
    ],
    show_tool_calls=True,
    markdown=True,
)

# Financial Agent
financial_agent = Agent(
    name="Finance AI Agent",
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[
        YFinanceTools(
            stock_price=True,
            stock_fundamentals=True,
            analyst_recommendations=True,
            company_news=True,
        )
    ],
    instructions=[
        "1. Use tables to display stock data for better readability.",
        "2. Always include the following details for any stock:",
        "   - Current price",
        "   - Key fundamentals (e.g., P/E ratio, market cap)",
        "   - Latest analyst recommendations (Buy/Hold/Sell)",
        "   - Recent company news (last 7 days)",
        "3. If data is unavailable, explain why and suggest alternative tools or sources.",
        "4. Keep the response concise and avoid unnecessary details.",
    ],
    show_tool_calls=True,
    markdown=True,
)

# Multi-AI Agent
multi_ai_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    team=[web_search_agent, financial_agent],
    instructions=[
        "1. Use the web search agent to find the latest financial news and updates.",
        "2. Use the financial agent to analyze stock data and provide structured insights.",
        "3. Always combine the results from both agents into a single, cohesive response.",
        "4. Follow these formatting guidelines:",
        "   - Use headings to separate sections (e.g., 'Latest News', 'Stock Analysis').",
        "   - Use tables for numerical data.",
        "   - Include clickable links for all sources.",
        "5. If the user's query is unclear, ask for clarification before proceeding.",
    ],
    show_tool_calls=True,
    markdown=True,
)

# print(response)
# print('************************************')
# print(type(response))
# print('************************************')
# print(type(result))


# Streamlit UI
st.title("Financial Insights AI")

st.write("Enter a stock ticker or company name to get financial analysis and the latest news.")

# User input
user_input = st.text_input("Enter stock ticker (e.g., NVDA, AAPL):", "")

if st.button("Analyze Stock"):
    if user_input:
        st.write("Generating insights, please wait...")
        
        # Run agent
        response = multi_ai_agent.run(f"Summarize analyst recommendations and share the latest news for {user_input}.")
        
        # Print response structure to debug (optional)
        # print(response)  # Check available attributes
        
        # Extract and display response content
    #     st.markdown(pprint_run_response(response, markdown=True), unsafe_allow_html=True)
    # else:
    #     st.warning("Please enter a stock ticker to proceed.")

    # else:
    #     st.write('Please enter valid input')
        
        if response:
            result = llm_model.generate_content(str(response))
            st.markdown(result.text, unsafe_allow_html=True)
        else:
            st.error("No valid response received. Try again with a different stock ticker.")




        # print(response)
        # print(dir(response))  # Check available attributes


        # formatted_response = pprint_run_response(response, markdown=True)

        # if formatted_response:
        #     st.markdown(formatted_response, unsafe_allow_html=True)
        # else:
        #     st.error("No response received. Please try again.")
