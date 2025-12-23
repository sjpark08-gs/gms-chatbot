import streamlit as st
import pandas as pd
import os
import re
import plotly.io as pio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# =========================================================
# 1. Basic Setup
# =========================================================
load_dotenv()
st.set_page_config(
    page_title="GMS Data Analysis Chatbot (Pro)",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# 2. Smart Excel Loader & Preprocess
# =========================================================
@st.cache_data 
def load_and_process_data(file):
    # 1. Load Smartly (Header Detection)
    preview_df = pd.read_excel(file, sheet_name="2nd treatment", header=None, nrows=10)
    header_idx = 0
    keywords = ['date', '일자', '날짜', 'period', 'time']

    for idx, row in preview_df.iterrows():
        row_str = row.astype(str).str.lower().tolist()
        if any(k in cell for cell in row_str for k in keywords):
            header_idx = idx
            break

    df = pd.read_excel(file, sheet_name="2nd treatment", header=header_idx)

    # 2. Preprocess
    df.columns = df.columns.astype(str).str.strip().str.replace('\n', '')

    for col in df.columns:
        if col.lower() in ['date', 'time', 'period', '일자', '날짜']:
            df.rename(columns={col: 'Date'}, inplace=True)
            break

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Year'] = df['Date'].dt.year  # ⭐ 미리 생성

    for col in df.columns:
        if col not in ['Date'] and df[col].dtype == 'object':
            try:
                df[col] = (
                    df[col].astype(str)
                    .str.replace(r'[$,\s]', '', regex=True)
                )
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass

    return df

# =========================================================
# 3. Column Definitions
# =========================================================
COLUMN_DEFINITIONS = {
    "Revenue": "Revenue, Sales, Income, Total Sales",
    "Penalty": "Penalty, Fine",
    "Water Consumption": "Water Usage, Water Consumption, Water",
    "Power Consumption": "Power Usage, Electricity, Energy",
    "Pipe fee": "Pipe Usage, Pipe Fee, Pipeline Cost",
    "Cehmical Consumption-GMS": "Chemical Usage, GMS Chemicals",
    "Cehmical Consumption-KE": "KE Chemical Usage, KE Chemicals",
    "Cost": "Cost, Expense, Expenditure",
    "Gross Profit": "Gross Profit, Operating Profit",
    "Net Profit": "Net Profit, Net Income",
    "GA": "GA, General Administrative Expenses"
}

DEFAULT_FILE_PATH = "Updated_Monthly_Report.xlsx"

# =========================================================
# 4. Token Optimizer
# =========================================================
def extract_years(text):
    years = re.findall(r"(20\d{2})", text)
    return list(set(map(int, years)))

def filter_columns_by_question(df, question):
    selected = ["Date", "Year"] 
    q = question.lower()

    for col, desc in COLUMN_DEFINITIONS.items():
        if any(word.strip().lower() in q for word in desc.split(",")):
            if col in df.columns:
                selected.append(col)

    return df[selected] if len(selected) > 2 else df

# =========================================================
# 5. UI Setup
# =========================================================
st.title("🤖 GMS Data Analysis Agent (Optimized)")

with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

source_file = uploaded_file if uploaded_file else DEFAULT_FILE_PATH

# =========================================================
# 6. Load Data
# =========================================================
if source_file:
    try:
        if "df" not in st.session_state:
             st.session_state.df = load_and_process_data(source_file)
    except Exception as e:
        st.error(f"File Load Error: {e}")
        st.stop()
else:
    st.error("No file available.")
    st.stop()

df = st.session_state.df

with st.expander("📊 Data Preview"):
    st.dataframe(df.head())

# =========================================================
# 7. Chat Interface
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "plot_json" in msg:
            st.plotly_chart(pio.from_json(msg["plot_json"]), use_container_width=True)

# =========================================================
# 8. Agent & Logic (언어 및 차트 생성 기능 강화)
# =========================================================
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0
)

# 딕셔너리를 문자열로 변환 (Prompt용)
column_def_str = "\n".join([f"- {k}: {v}" for k, v in COLUMN_DEFINITIONS.items()])

# 에이전트에게 주는 핵심 지시사항
PREFIX_PROMPT = f"""
You are a Senior Data Analyst.
Use ONLY the provided DataFrame.

[Column Mapping]
Use this to interpret user queries:
{column_def_str}

[Instructions]
1. **Language Policy:** - Detect the language of the user's question.
   - If the user asks in English, you MUST respond in English.
   - If the user asks in Korean, you MUST respond in Korean.
   
2. **Visualization/Charting:**
   - If the user asks to draw a chart or visualize data, use the `plotly` library.
   - **CRITICAL:** After creating a figure (e.g., `fig`), you MUST save it as a JSON file named 'output_plot.json' using the code: `fig.write_json('output_plot.json')`.
   - Do not just mention the chart; execute the code to save the file.

3. **Tool Usage:** You have a pandas dataframe tool. Use it to run python code.
4. **Verification:** Check column names before using them.
5. **Final Answer:** Provide a concise summary of your findings along with the requested data.
"""

if prompt := st.chat_input("Ask a question..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                # 1. Filter Data
                years = extract_years(prompt)
                df_filtered = df.copy()

                if years:
                    df_filtered = df_filtered[df_filtered["Year"].isin(years)]
                
                if df_filtered.empty:
                     df_filtered = df.copy()

                df_filtered = filter_columns_by_question(df_filtered, prompt)

                # 2. Create Agent
                agent = create_pandas_dataframe_agent(
                    llm,
                    df_filtered,
                    verbose=True,
                    allow_dangerous_code=True,
                    prefix=PREFIX_PROMPT,
                    agent_type="openai-tools",
                    max_iterations=10,
                    agent_executor_kwargs={
                        "handle_parsing_errors": True
                    }
                )

                # 에이전트 실행
                response = agent.invoke({"input": prompt})
                answer = response["output"]

                st.markdown(answer)
                msg_data = {"role": "assistant", "content": answer}

                # 3. 차트 파일 확인 및 출력
                if os.path.exists("output_plot.json"):
                    with open("output_plot.json", "r", encoding="utf-8") as f:
                        plot_json_content = f.read()
                    
                    # 화면에 차트 표시
                    fig = pio.from_json(plot_json_content)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 메시지 기록에 저장
                    msg_data["plot_json"] = plot_json_content
                    # 사용 후 파일 삭제 (다음 질문과의 혼선 방지)
                    os.remove("output_plot.json")

                st.session_state.messages.append(msg_data)

            except Exception as e:
                st.warning(f"⚠️ Analysis failed. (Error: {e})")
