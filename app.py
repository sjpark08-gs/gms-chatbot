import streamlit as st
import pandas as pd
import os
import plotly.io as pio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# 1. Basic Setup
load_dotenv()
st.set_page_config(page_title="GMS Data Analysis Chatbot", page_icon="📊")

# ==========================================
# [Developer Settings]
# 1. Column Dictionary (Translated Values to English)
# Note: Keep the Keys (Left side) exactly matching your Excel Headers!
COLUMN_DEFINITIONS = {
    "Date": "Date, Time, Period, Duration",
    "Revenue": "Revenue, Sales, Income, Total Sales",
    "Panelty": "Penalty, Fine",
    "Water Consumption" : "Water Usage, Water Consumption, Water",
    "Power Consumption" : "Power Usage, Electricity, Energy, Electric Bill",
    "Pipe fee" : "Pipe Usage, Pipe Fee, Pipeline Cost",
    "Cehmical Consumption-GMS" : "Chemical Usage, Chemical Cost, GMS Chemicals",
    "Cehmical Consumption-KE" : "KE Chemical Usage, KEI Chemicals, KEI Cost",
    "Cost": "Cost, Expense, Expenditure, Spending",
    "Gross Profit": "Gross Profit, Operating Profit, Margin",
    "Net Profit": "Net Profit, Net Income, Bottom Line",
    "GA" : "GA, General Administrative Expenses, Admin Cost"
}

# 2. Default File Path (Must exist in GitHub repo)
DEFAULT_FILE_PATH = "Updated_Monthly_Repor.xlsx"
# ==========================================

st.title("🤖 GMS Excel Data Analysis Chatbot")

# Sidebar Settings
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload New Excel File (Optional)", type=["xlsx", "xls"])
    
    with st.expander("ℹ️ View Column Definitions"):
        st.json(COLUMN_DEFINITIONS)
    
    st.markdown("---")
    st.markdown("**Tips:**\n- 'Show me the monthly revenue trend.'\n- 'Which region has the highest cost?'")

# File Selection Logic
target_file = None

if uploaded_file is not None:
    # Priority 1: User uploaded file
    target_file = uploaded_file
    st.toast("📂 Analyzing the uploaded file.", icon="✅")
elif os.path.exists(DEFAULT_FILE_PATH):
    # Priority 2: Default file (GitHub)
    target_file = DEFAULT_FILE_PATH
    st.toast(f"📂 Analyzing the default data ('{DEFAULT_FILE_PATH}').", icon="ℹ️")
else:
    # No file found
    st.error(f"Error: Could not find the default file ('{DEFAULT_FILE_PATH}') and no file was uploaded.")
    st.stop()


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "plot_json" in message:
            fig = pio.from_json(message["plot_json"])
            st.plotly_chart(fig, use_container_width=True)

# Main Logic (When target_file is confirmed)
if target_file:
    try:
        xls = pd.ExcelFile(target_file)
        target_sheet = "2nd treatment"

        if target_sheet in xls.sheet_names:
            df = pd.read_excel(target_file, sheet_name=target_sheet)
            # Data Preview
            file_label = "Uploaded Data" if uploaded_file else "Default Data"
            with st.expander(f"📊 {file_label} Preview ({target_sheet})"):
                st.dataframe(df.head())
        else:
            st.error(f"Error: The sheet '{target_sheet}' was not found in the Excel file.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        st.stop()

    # LLM Setup
    # Using gpt-4o as requested. Use "gpt-4o-mini" to save costs if needed.
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True, 
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True} 
    )

    if prompt := st.chat_input("Ask a question..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    # History Context (Last 2 messages)
                    chat_history_text = ""
                    for msg in st.session_state.messages[-2:]: 
                        role = "User" if msg["role"] == "user" else "AI"
                        content = msg["content"]
                        chat_history_text += f"{role}: {content}\n"

                    # Updated Instruction for English Output
                    instruction = f"""
                    You are a capable data analyst.
                    
                    [Data Column Definitions]
                    {COLUMN_DEFINITIONS}
                    Refer to the definitions above to match user queries to the correct columns.
                    
                    [Chart Generation Rules]
                    1. Use **Plotly Express** (variable name: fig).
                    2. Save the graph as a JSON file (`output_plot.json`).
                    3. Do NOT use `fig.show()`.
                    
                    [Context from Previous Conversation]
                    {chat_history_text}
                    
                    **Please answer the question in English.**
                    """
                    full_prompt = f"{instruction}\n\n[Current Question]\n{prompt}"
                    response = agent.invoke(full_prompt)
                    answer = response['output']

                    st.markdown(answer)
                    msg_data = {"role": "assistant", "content": answer}

                    if os.path.exists("output_plot.json"):
                        try:
                            with open("output_plot.json", "r", encoding="utf-8") as f:
                                plot_json = f.read()
                        except UnicodeDecodeError:
                            with open("output_plot.json", "r", encoding="cp949") as f:
                                plot_json = f.read()
                        
                        fig = pio.from_json(plot_json)
                        st.plotly_chart(fig, use_container_width=True)
                        msg_data["plot_json"] = plot_json
                        os.remove("output_plot.json")

                    st.session_state.messages.append(msg_data)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
