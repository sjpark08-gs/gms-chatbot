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
st.set_page_config(page_title="GMS Data Analysis Pro", page_icon="📊", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None

DEFAULT_FILE_PATH = "Updated_Monthly_Report.xlsx"

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

# =========================================================
# 2. Smart Excel Loader & Preprocess
# =========================================================
@st.cache_data 
def load_and_process_data(file):
    try:
        preview_df = pd.read_excel(file, sheet_name="2nd treatment", header=None, nrows=10)
        header_idx = 0
        keywords = ['date', '일자', '날짜', 'period', 'time']
        for idx, row in preview_df.iterrows():
            row_str = row.astype(str).str.lower().tolist()
            if any(k in cell for cell in row_str for k in keywords):
                header_idx = idx
                break

        df = pd.read_excel(file, sheet_name="2nd treatment", header=header_idx)
        df.columns = df.columns.astype(str).str.strip().str.replace('\n', '')

        for col in df.columns:
            if col.lower() in ['date', 'time', 'period', '일자', '날짜']:
                df.rename(columns={col: 'Date'}, inplace=True)
                break

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            
        for col in df.columns:
            if col not in ['Date', 'Year', 'Month'] and df[col].dtype == 'object':
                try:
                    df[col] = df[col].astype(str).str.replace(r'[$,\s]', '', regex=True)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except: pass
        return df
    except Exception as e:
        st.error(f"File Load Error: {e}")
        return None
    
# =========================================================
# 3. Sidebar (English UI)
# =========================================================
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

    # --- 📋 Available Columns ---
    st.divider()
    st.subheader("📋 Available Columns")
    raw_cols = "Date	Revenue	Panelty	Net Revenue	Water Consumption	Power Consumption	Pipe fee	Chemical Consumption-GMS	Operation Cost-GMS	Chemical Consumption-KE	Operation Cost-KE	Total COGS	Gross Profit	Financial Income	Financial Expense	Sales Expense	GA	Net Profit	Net Profit before Tax	CIT	Net Profit after Tax	Revenue (Quantity)	Panelty (Quantity)	Water Consumption (Quantity)	Power Consumption (Quantity)	Other Income(Claim)"
    column_list = [c.strip() for c in raw_cols.split('\t') if c.strip()]

    with st.expander("View Column List", expanded=False):
        for col in column_list:
            st.caption(f"• {col}")

    # --- 🧮 Calculation Guide ---
    st.divider()
    st.subheader("🧮 Calculation Formulas")
    formulas = [
        "**Net Revenue** = Revenue + Penalty + Other Income",
        "**Operation Cost** = Water Cons. + Power Cons. + Pipe fee",
        "**Total COGS** = Operation Cost-GMS + Operation Cost-KE",
        "**Gross Profit** = Net Revenue - Total COGS",
        "**Net Profit** = Gross Profit + Fin. Inc - Fin. Exp - Sales Exp - GA"
    ]
    for f in formulas:
        st.caption(f)

    # --- ℹ️ Notice ---
    st.divider()
    st.subheader("ℹ️ Notice")
    st.info("""
    - Each Chemical Cost is included in the Operation Cost.
    - The operation cost does not include sludge disposal fees, monitoring costs, or other operating expenses.
    - 'Chemical Consumption' refers to **Cost (Value)**; 'Usage Volume' is not available.
    - Please use the words in Column list, when you chat with the bot.
    """)

    # --- 📝 Quick Memo ---
    st.divider()
    st.subheader("📝 Quick Memo")
    memo_input = st.text_area("Message:", height=100)
    
    if st.button("💾 Save"):
        if memo_input.strip():
            import datetime
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            with open("memos.txt", "a", encoding="utf-8") as f:
                f.write(f"[{now}] {memo_input}\n---\n")
            st.rerun()

    if os.path.exists("memos.txt"):
        with st.expander("📖 Saved Memos"):
            try:
                with open("memos.txt", "r", encoding="utf-8", errors='replace') as f:
                    st.text(f.read())
            except:
                st.text("Unable to load memos.")

# =========================================================
# 4. Helpers
# =========================================================
def extract_years(text):
    years = re.findall(r"(20\d{2})", text)
    return list(set(map(int, years)))

def filter_columns_by_question(df, question):
    selected = ["Date", "Year"] 
    q = question.lower()
    for col, desc in COLUMN_DEFINITIONS.items():
        if any(word.strip().lower() in q for word in desc.split(",")):
            if col in df.columns: selected.append(col)
    return df[selected] if len(selected) > 2 else df

# =========================================================
# 5. UI Setup & Data Load
# =========================================================
st.title("🤖 GMS Data Analysis Agent")

source_file = uploaded_file if uploaded_file else DEFAULT_FILE_PATH

if st.session_state.df is None or uploaded_file:
    st.session_state.df = load_and_process_data(source_file)

df = st.session_state.df

if df is not None:
    with st.expander("✅ Data Loading Status", expanded=False):
        st.write("Current data availability by Year and Month:")
        inventory = df.groupby(['Year', 'Month']).size().unstack(fill_value=0)
        st.table(inventory)
else:
    st.stop()

with st.expander("📊 Data Preview"):
    st.dataframe(df.head())

# =========================================================
# 7. Chat Interface
# =========================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "plot_json" in msg:
            st.plotly_chart(pio.from_json(msg["plot_json"]), use_container_width=True)

# =========================================================
# 8. Agent & Logic (차트 생성 억제 로직 강화)
# =========================================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
column_def_str = "\n".join([f"- {k}: {v}" for k, v in COLUMN_DEFINITIONS.items()])

if prompt := st.chat_input("Ask a question..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    chart_keywords = ["차트", "그래프", "그려", "시각화", "chart", "graph", "plot", "visualize"]
    is_chart_requested = any(word in prompt.lower() for word in chart_keywords)

    history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:-1]])

    PREFIX_PROMPT = f"""
    You are a Senior Data Analyst. Use ONLY the provided DataFrame.

    [Column Mapping]
    {column_def_str}

    [Conversation Context]
    {history}
    

    [Relationship of Each Column / Calculation Formulas]
    Follow these rules for calculations:
    1. Net Revenue = Revenue + Panelty + Other Income(Claim)
    2. Operation Cost = Water Consumption + Power Consumption + Pipe fee
    3. Total COGS = Operation Cost-GMS + Operation Cost-KE
    4. Gross Profit = Net Revenue - Total COGS
    5. Net Profit = Gross Profit + Financial Income - Financial Expense - Sales Expense - GA

    [Instructions]
    1. **Language Policy:** Detect language and respond in the SAME language.
    2. **Efficiency:** When calculating for multiple months, create new temporary columns in the dataframe for these formulas first, then summarize. This is faster than calculating row by row.
    3. **Verification:** Check column names before using them.
    4. **CRITICAL (Show All Months):** When asked for a year's data, you MUST provide values for all 12 months.
    5. **CRITICAL (No Scaling):** DO NOT multiply or divide values by 1,000. Use raw numbers in df.
    6. **Final Answer:** Provide a full table and a concise summary.
    7. Show unit automatically. Most of items are 'VND'. but, Revenue (Quantity) and Panelty (Quantity), Water Consumption (Quantity) are "m3", Power Consumption (Quantity) is "kWh" 

    [Strict Visualization Rules]
    - NEVER create a chart unless the user explicitly uses words like 'chart', 'graph', or 'draw a chart'.
    - If a chart is requested, save as 'output_plot.json' via `fig.write_json('output_plot.json')`.
    """

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                # 데이터 필터링
                years = extract_years(prompt)
                df_filtered = df.copy()
                if years:
                    df_filtered = df_filtered[df_filtered["Year"].isin(years)]
                
                # 에이전트 생성 (사고 횟수 상향)
                agent = create_pandas_dataframe_agent(
                    llm, df_filtered, verbose=True, allow_dangerous_code=True,
                    prefix=PREFIX_PROMPT, agent_type="openai-tools", 
                    max_iterations=20 # 복잡한 수식 처리를 위해 상향 유지
                )

                # 명령어 구성
                if is_chart_requested:
                    final_input = f"{prompt} (Instruction: Chart requested. Save output_plot.json)"
                else:
                    final_input = f"{prompt} (Instruction: TEXT TABLE only. No charts allowed.)"

                response = agent.invoke({"input": final_input})
                answer = response["output"]

                st.markdown(answer)
                msg_data = {"role": "assistant", "content": answer}

                # 차트 출력 처리
                if os.path.exists("output_plot.json"):
                    if is_chart_requested:
                        with open("output_plot.json", "r", encoding="utf-8", errors='replace') as f:
                            plot_json_content = f.read()
                        fig = pio.from_json(plot_json_content)
                        st.plotly_chart(fig, use_container_width=True)
                        msg_data["plot_json"] = plot_json_content
                    os.remove("output_plot.json")

                st.session_state.messages.append(msg_data)

            except Exception as e:
                if "max iterations" in str(e).lower():
                    st.warning("⚠️ Query is too complex. Please try breaking it down.")
                else:
                    st.warning(f"⚠️ Analysis failed. (Error: {e})")
