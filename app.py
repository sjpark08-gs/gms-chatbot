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
# 8. Agent & Logic (대화 기억 기능 추가)
# =========================================================
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0
)

# 1. 이전 대화 기록 요약본 생성 (에이전트에게 문맥 전달용)
# 너무 길어질 경우 최근 5~10개만 유지하는 것이 토큰 절약에 좋습니다.
chat_history_str = ""
for msg in st.session_state.messages[-6:]:  # 최근 3번의 대화쌍(총 6개 메시지) 참조
    role_label = "User" if msg["role"] == "user" else "Assistant"
    chat_history_str += f"{role_label}: {msg['content']}\n"

# 딕셔너리를 문자열로 변환 (Prompt용)
column_def_str = "\n".join([f"- {k}: {v}" for k, v in COLUMN_DEFINITIONS.items()])

# 2. 프롬프트에 [Conversation History] 섹션 추가
PREFIX_PROMPT = f"""
You are a Senior Data Analyst.
Use ONLY the provided DataFrame.

[Column Mapping]
{column_def_str}

[Conversation History]
{chat_history_str}
(참고: 위 내용은 이전 대화입니다. 사용자의 질문이 '그것', '저번' 등 이전 문맥을 포함하면 위 내용을 참고하세요.)

[Instructions]
1. **Language Policy:** Detect the language of the user's question and respond in the same language.
2. **Visualization:** If requested, use plotly and MUST save it as 'output_plot.json' via `fig.write_json('output_plot.json')`.
3. **Tool Usage:** Use the pandas dataframe tool for analysis.
4. **Verification:** Check column names before using them.
"""

if prompt := st.chat_input("Ask a question..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                # 데이터 필터링 로직 (기존과 동일)
                years = extract_years(prompt)
                df_filtered = df.copy()
                if years:
                    df_filtered = df_filtered[df_filtered["Year"].isin(years)]
                if df_filtered.empty:
                    df_filtered = df.copy()
                df_filtered = filter_columns_by_question(df_filtered, prompt)

                # 에이전트 생성 (매번 현재 필터링된 데이터와 문맥을 가지고 생성)
                agent = create_pandas_dataframe_agent(
                    llm,
                    df_filtered,
                    verbose=True,
                    allow_dangerous_code=True,
                    prefix=PREFIX_PROMPT,
                    agent_type="openai-tools",
                    max_iterations=10,
                    agent_executor_kwargs={"handle_parsing_errors": True}
                )

                response = agent.invoke({"input": prompt})
                answer = response["output"]

                st.markdown(answer)
                msg_data = {"role": "assistant", "content": answer}

                # 차트 출력 로직 (기존과 동일)
                if os.path.exists("output_plot.json"):
                    with open("output_plot.json", "r", encoding="utf-8") as f:
                        plot_json_content = f.read()
                    st.plotly_chart(pio.from_json(plot_json_content), use_container_width=True)
                    msg_data["plot_json"] = plot_json_content
                    os.remove("output_plot.json")

                st.session_state.messages.append(msg_data)

            except Exception as e:
                st.warning(f"⚠️ Analysis failed. (Error: {e})")

