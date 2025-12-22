import streamlit as st
import pandas as pd
import os
import plotly.io as pio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# 1. 기본 설정
load_dotenv()
st.set_page_config(page_title="GMS 데이터 분석 챗봇", page_icon="📊")

# ==========================================
# [개발자 설정 구역]
# 1. 컬럼 사전
COLUMN_DEFINITIONS = {
"Date": "날짜, 일자, 시기, 기간",

"Revenue": "매출, 수익, 수입, 매출액",

"Panelty": "패널티, 벌금",

"Water Consumption" : "물사용량, 물 사용량",

"Power Consumption" : "전력량, 전기사용량, 전기요금, 전기세",

"Pipe fee" : "파이프사용량, 파이프요금, 파이프 사용량, 파이프 요금",

"Cehmical Consumption-GMS" : "약품, 약품사용량, 약품 사용량, 약품비용, 약품비, GMS 약품사용량",

"Cehmical Consumption-KE" : "KE 약품 사용량, KEI 약품사용량, KEI약품사용량, KEI 약품비, KEI약품비",

"Cost": "비용, 지출, 원가",

"Gross Profit": "영업이익, 마진",

"Net Profit": "순수익, 순이익, 당기순이익",

"GA" : "일반관리비"
}

# 2. 기본 파일 설정 (GitHub에 이 파일이 꼭 있어야 합니다!)
DEFAULT_FILE_PATH = "Updated_Monthly_Report.xlsx"
# ==========================================

st.title("🤖 GMS 엑셀 데이터 분석 챗봇")

# 사이드바 설정
with st.sidebar:
    st.header("설정")
    uploaded_file = st.file_uploader("새로운 엑셀 파일 업로드 (선택사항)", type=["xlsx", "xls"])
    
    with st.expander("ℹ️ 등록된 컬럼 사전 보기"):
        st.json(COLUMN_DEFINITIONS)
    
    st.markdown("---")
    st.markdown("**사용 팁:**\n- '월별 매출 추이를 그려줘'")

# 분석할 파일 결정 로직
target_file = None

if uploaded_file is not None:
    # 1순위: 사용자가 업로드한 파일
    target_file = uploaded_file
    st.toast("📂 사용자가 업로드한 파일을 분석합니다.", icon="✅")
elif os.path.exists(DEFAULT_FILE_PATH):
    # 2순위: 기본 파일 (GitHub에 있는 파일)
    target_file = DEFAULT_FILE_PATH
    st.toast(f"📂 기본 데이터('{DEFAULT_FILE_PATH}')를 분석합니다.", icon="ℹ️")
else:
    # 파일이 아예 없는 경우
    st.error(f"오류: 기본 파일('{DEFAULT_FILE_PATH}')을 찾을 수 없고, 업로드된 파일도 없습니다.")
    st.stop()


# 세션 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "plot_json" in message:
            fig = pio.from_json(message["plot_json"])
            st.plotly_chart(fig, use_container_width=True)

# 메인 로직 실행 (target_file이 확정된 상태)
if target_file:
    try:
        xls = pd.ExcelFile(target_file)
        target_sheet = "2nd treatment"

        if target_sheet in xls.sheet_names:
            df = pd.read_excel(target_file, sheet_name=target_sheet)
            # 데이터 미리보기 (기본 파일일 때와 업로드일 때 문구 다르게)
            file_label = "업로드된 데이터" if uploaded_file else "기본 데이터"
            with st.expander(f"📊 {file_label} 미리보기 ({target_sheet})"):
                st.dataframe(df.head())
        else:
            st.error(f"오류: '{target_sheet}' 시트가 없습니다.")
            st.stop()
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        st.stop()

    # LLM 설정
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True, 
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True} 
    )

    if prompt := st.chat_input("질문을 입력하세요..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("분석 중입니다..."):
                try:
                    # 1. 이전 대화 기록을 텍스트로 변환 (최근 4개 메시지만 참조 - 토큰 절약)
                    chat_history_text = ""
                    for msg in st.session_state.messages[-4:]: 
                        role = "User" if msg["role"] == "user" else "AI"
                        content = msg["content"]
                        chat_history_text += f"{role}: {content}\n"

                    # 2. 프롬프트 구성 (지시사항 + 대화기록 + 현재질문)
                    instruction = f"""
                    너는 유능한 데이터 분석가야.
                    
                    [데이터 컬럼 명세서]
                    {COLUMN_DEFINITIONS}
                    사용자가 한글로 질문하면 위 명세서를 참고해.
                    
                    [차트 그리기 규칙]
                    1. **Plotly Express** 사용 (변수명: fig)
                    2. 그래프를 JSON 파일로 저장 (`output_plot.json`)
                    3. `fig.show()` 금지
                    
                    [기억해야 할 이전 대화]
                    아래 대화의 맥락을 파악해서 현재 질문에 답해. 
                    특히 "이걸로", "바꿔줘", "다시 그려줘" 같은 지시가 나오면 이전 대화의 데이터를 기반으로 수정해.
                    ---
                    {chat_history_text}
                    ---
                    
                    최종 답변은 한국어로 해줘.
                    """
                    
                    # 3. 질문 전달
                    full_prompt = f"{instruction}\n\n[현재 질문]\n{prompt}"
                    
                    response = agent.invoke(full_prompt)
                    answer = response['output']

                    # ... (이하 코드는 기존과 동일: 답변 출력, 그래프 처리 등) ...
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
                    st.error(f"오류가 발생했습니다: {e}")
    st.info("👈 파일을 업로드해주세요.")


