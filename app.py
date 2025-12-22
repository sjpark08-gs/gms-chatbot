import streamlit as st
import pandas as pd
import os
import plotly.io as pio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# 1. Basic Setup & Load Environment Variables
load_dotenv()
st.set_page_config(page_title="GMS Data Analysis Chatbot", page_icon="📊")

# ==========================================
# [Developer Settings] Column Dictionary
# ==========================================
COLUMN_DEFINITIONS = {
    "Date": "Date, Time, Period",
    "Revenue": "Revenue, Sales, Income",
    "Panelty": "Penalty, Fine",
    "Water Consumption": "Water Usage, Water",
    "Power Consumption": "Electricity Usage, Power Usage, Electric Bill",
    "Pipe fee": "Pipe Usage, Pipe Fee",
    "Cehmical Consumption-GMS": "Chemical Usage, Chemical Cost, GMS Chemicals",
    "Cehmical Consumption-KE": "KE Chemical Usage, KEI Chemicals, KEI Cost",
    "Cost": "Cost, Expense, Expenditure",
    "Gross Profit": "Operating Profit, Margin, Gross Profit",
    "Net Profit": "Net Income, Net Profit, Bottom line",
    "GA": "General Administrative Expenses, GA"
}
# ==========================================

st.title("🤖 GMS Excel Data Analysis Chatbot")
st.markdown("Upload an Excel file and ask questions! (Analyzing '2nd treatment' sheet)")

with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
    
    with st.expander("ℹ️ View Column Definitions"):
        st.json(COLUMN_DEFINITIONS)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Draw Plotly Graph if available
        if "plot_json" in message:
            fig = pio.from_json(message["plot_json"])
            st.plotly_chart(fig, use_container_width=True)

if uploaded_file is not None:
    try:
        xls = pd.ExcelFile(uploaded_file)
        target_sheet = "2nd treatment"

        if target_sheet in xls.sheet_names:
            df = pd.read_excel(uploaded_file, sheet_name=target_sheet)
            with st.expander(f"📊 Data Preview ('{target_sheet}')"):
                st.dataframe(df.head())
        else:
            st.error(f"Error: The sheet '{target_sheet}' was not found in the Excel file.")
            st.stop()

    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # LLM Setup (Corrected model name for stability)
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
                    # 1. Convert chat history to text (last 4 messages)
                    chat_history_text = ""
                    for msg in st.session_state.messages[-4:]: 
                        role = "User" if msg["role"] == "user" else "AI"
                        content = msg["content"]
                        chat_history_text += f"{role}: {content}\n"

                    # 2. Construct Prompt (English Instructions)
                    instruction = f"""
                    You are a capable data analyst.
                    
                    [Data Column Definitions]
                    {COLUMN_DEFINITIONS}
                    Refer to the definitions above to map user queries to the correct columns.
                    
                    [Chart Generation Rules]
                    1. Use **Plotly Express** (variable name: fig).
                    2. Save the graph as a JSON file (`output_plot.json`).
                    3. Do NOT use `fig.show()`.
                    4. Use `template='plotly_white'` for a clean look.
                    5. Add a clear title to the chart.
                    
                    [Context from Previous Conversation]
                    Understand the context below to answer the current question.
                    Especially if the user says "Change this to...", "Redraw as...", use the previous data.
                    ---
                    {chat_history_text}
                    ---
                    
                    **Please provide the final answer in English.**
                    """
                    
                    # 3. Send Request
                    full_prompt = f"{instruction}\n\n[Current Question]\n{prompt}"
                    
                    response = agent.invoke(full_prompt)
                    answer = response['output']

                    # Output logic
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
else:
    st.info("👈 Please upload an Excel file to start.")
