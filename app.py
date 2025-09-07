# BUSINESS SCIENCE
# Pandas Data Analyst App
# -----------------------

# This app is designed to help you analyze data and create data visualizations from natural language requests.

# Imports
# !pip install git+https://github.com/business-science/ai-data-science-team.git --upgrade

from openai import OpenAI

import streamlit as st
import pandas as pd
import plotly.io as pio
import json

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI

from ai_data_science_team import (
    PandasDataAnalyst,
    DataWranglingAgent,
    DataVisualizationAgent,
)


# * APP INPUTS ----

MODEL_LIST = ["gpt-4o-mini", "gpt-4o"]
TITLE = "Analise de Dados com IA"

# ---------------------------
# Streamlit App Configuration
# ---------------------------

st.set_page_config(
    page_title=TITLE,
    page_icon="📊",
)
st.title(TITLE)


with st.expander("Exemplos", expanded=False):
    st.write(
        """
            Conjunto de Dados de Bicicletas:

             - Mostre os 5 principais modelos de bicicleta por vendas estendidas.

             - Mostre os 5 principais modelos de bicicleta por vendas estendidas em um gráfico de barras.

             - Mostre os 5 principais modelos de bicicleta por vendas estendidas em um gráfico de pizza.

             - Faça um gráfico de vendas estendidas por mês para cada modelo de bicicleta. Use cores para identificar os modelos de bicicleta.
        """
    )

# ---------------------------
# OpenAI API Key Entry and Test
# ---------------------------

st.sidebar.header("Insira sua chave API")

st.session_state["OPENAI_API_KEY"] = st.sidebar.text_input(
    "API Key",
    type="password",
    help="Sua Chave de API da OpenAI é necessária para iniciar o aplicativo.",
)

# Test OpenAI API Key
if st.session_state["OPENAI_API_KEY"]:
    # Set the API key for OpenAI
    client = OpenAI(api_key=st.session_state["OPENAI_API_KEY"])

    # Test the API key (optional)
    try:
        # Example: Fetch models to validate the key
        models = client.models.list()
        st.success("Chave API é valida!")
    except Exception as e:
        st.error(f"Chave API invalida: {e}")
else:
    st.info("Por favor, insira sua Chave de API da OpenAI para continuar.")
    st.stop()


# * OpenAI Model Selection

model_option = st.sidebar.selectbox("Escolha o Modelo OpenAI", MODEL_LIST, index=0)

llm = ChatOpenAI(model=model_option, api_key=st.session_state["OPENAI_API_KEY"])


# ---------------------------
# File Upload and Data Preview
# ---------------------------

st.markdown("""
Carregue um arquivo CSV ou Excel e faça perguntas sobre seus dados.
O agente de IA analisará seu conjunto de dados e retornará tabelas de dados ou gráficos interativos.
""")

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file", type=["csv", "xlsx", "xls"]
)
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        # Tenta primeiro com encoding UTF-8 e separador ;
        try:
            df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
        except UnicodeDecodeError:
            # Se falhar, tenta com encoding Latin-1 comum em arquivos em português
            try: 
                uploaded_file.seek(0)  # Volta ao início do arquivo
                df = pd.read_csv(uploaded_file, sep=';', encoding='latin-1')
            except UnicodeDecodeError:
                uploaded_file.seek(0)  # Volta ao início do arquivo
                df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())
else:
    st.info("Por favor, carregue um arquivo CSV ou Excel para iniciar.")
    st.stop()

# ---------------------------
# Initialize Chat Message History and Storage
# ---------------------------

msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("Como posso ajudá-lo?")

if "plots" not in st.session_state:
    st.session_state.plots = []

if "dataframes" not in st.session_state:
    st.session_state.dataframes = []


def display_chat_history():
    for msg in msgs.messages:
        with st.chat_message(msg.type):
            if "PLOT_INDEX:" in msg.content:
                plot_index = int(msg.content.split("PLOT_INDEX:")[1])
                st.plotly_chart(
                    st.session_state.plots[plot_index], key=f"history_plot_{plot_index}"
                )
            elif "DATAFRAME_INDEX:" in msg.content:
                df_index = int(msg.content.split("DATAFRAME_INDEX:")[1])
                st.dataframe(
                    st.session_state.dataframes[df_index],
                    key=f"history_dataframe_{df_index}",
                )
            else:
                st.write(msg.content)


# Render current messages from StreamlitChatMessageHistory
display_chat_history()

# ---------------------------
# AI Agent Setup
# ---------------------------

LOG = False

pandas_data_analyst = PandasDataAnalyst(
    model=llm,
    data_wrangling_agent=DataWranglingAgent(
        model=llm,
        log=LOG,
        bypass_recommended_steps=True,
        n_samples=100,
    ),
    data_visualization_agent=DataVisualizationAgent(
        model=llm,
        n_samples=100,
        log=LOG,
    ),
)

# ---------------------------
# Chat Input and Agent Invocation
# ---------------------------

if question := st.chat_input("Insira sua pergunta aqui:", key="query_input"):
    if not st.session_state["OPENAI_API_KEY"]:
        st.error("Por favor, insira sua Chave de API da OpenAI para continuar.")
        st.stop()

    with st.spinner("Analisando..."):
        st.chat_message("human").write(question)
        msgs.add_user_message(question)

        try:
            pandas_data_analyst.invoke_agent(
                user_instructions=question,
                data_raw=df,
            )
            result = pandas_data_analyst.get_response()
        except Exception as e:
            st.chat_message("ai").write(
                "Ocorreu um erro ao processar sua consulta. Por favor, tente novamente."
            )
            msgs.add_ai_message(
                "Ocorreu um erro ao processar sua consulta. Por favor, tente novamente."
            )
            st.stop()

        routing = result.get("routing_preprocessor_decision")

        if routing == "chart" and not result.get("plotly_error", False):
            # Process chart result
            plot_data = result.get("plotly_graph")
            if plot_data:
                # Convert dictionary to JSON string if needed
                if isinstance(plot_data, dict):
                    plot_json = json.dumps(plot_data)
                else:
                    plot_json = plot_data
                plot_obj = pio.from_json(plot_json)
                response_text = "Gráfico gerado."
                # Store the chart
                plot_index = len(st.session_state.plots)
                st.session_state.plots.append(plot_obj)
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"PLOT_INDEX:{plot_index}")
                st.chat_message("ai").write(response_text)
                st.plotly_chart(plot_obj)
            else:
                st.chat_message("ai").write("O agente não retornou um gráfico válido.")
                msgs.add_ai_message("O agente não retornou um gráfico válido.")

        elif routing == "table":
            # Process table result
            data_wrangled = result.get("data_wrangled")
            if data_wrangled is not None:
                response_text = "Tabela gerada."
                # Ensure data_wrangled is a DataFrame
                if not isinstance(data_wrangled, pd.DataFrame):
                    data_wrangled = pd.DataFrame(data_wrangled)
                df_index = len(st.session_state.dataframes)
                st.session_state.dataframes.append(data_wrangled)
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                st.chat_message("ai").write(response_text)
                st.dataframe(data_wrangled)
            else:
                st.chat_message("ai").write("O agente não retornou uma tabela válida.")
                msgs.add_ai_message("O agente não retornou uma tabela válido.")
        else:
            # Fallback if routing decision is unclear or if chart error occurred
            data_wrangled = result.get("data_wrangled")
            if data_wrangled is not None:
                response_text = (
                    "Houve um problema ao gerar o gráfico. " 
                    "Estou retornando a tabela de dados em seu lugar."
                    
                )
                if not isinstance(data_wrangled, pd.DataFrame):
                    data_wrangled = pd.DataFrame(data_wrangled)
                df_index = len(st.session_state.dataframes)
                st.session_state.dataframes.append(data_wrangled)
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                st.chat_message("ai").write(response_text)
                st.dataframe(data_wrangled)
            else:
                response_text = (
                    "Ocorreu um erro ao processar sua consulta. Por favor, tente novamente."
                )
                msgs.add_ai_message(response_text)
                st.chat_message("ai").write(response_text)
