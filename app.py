"""App de retenção — Terra Signal (Grupo 2).

O vendedor digita o customerID e vê, numa tela só, se o cliente vai fazer churn (Sim/Não),
os números principais e os serviços contratados. O modelo é o de modelo_churn.ipynb (limiar 0,20).

Rodar localmente:  python app.py   (ou: streamlit run app.py)
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Executado com "python app.py" (ou o botão Run do VS Code): inicia o servidor do Streamlit,
# que roda este mesmo arquivo e abre o app no navegador.
if __name__ == "__main__" and not st.runtime.exists():
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", __file__]
    sys.exit(stcli.main())

PASTA = Path(__file__).parent
ARQUIVO_MODELO = PASTA / "modelo_churn.joblib"
ARQUIVO_CARTEIRA = PASTA / "inference.csv"

CONTRATO = {"Month-to-month": "Mensal", "One year": "1 ano", "Two year": "2 anos"}
INTERNET = {"DSL": "DSL", "Fiber optic": "Fibra óptica", "No": "Sem internet"}
PAGAMENTO = {
    "Electronic check": "Cheque eletrônico",
    "Mailed check": "Cheque por correio",
    "Bank transfer (automatic)": "Transferência automática",
    "Credit card (automatic)": "Cartão automático",
}
ADICIONAIS = {
    "OnlineSecurity": "Segurança online",
    "TechSupport": "Suporte técnico",
    "OnlineBackup": "Backup online",
    "DeviceProtection": "Proteção de dispositivo",
    "StreamingTV": "Streaming de TV",
    "StreamingMovies": "Streaming de filmes",
}


# ==============================================================================================
# Preparação dos dados: os mesmos passos das seções 4 e 5 do modelo_churn.ipynb.
# O modelo salvo só recebe os dados depois deles.
# ==============================================================================================
def preparar_dados(df):
    df = df.copy()

    # 4.1. PhoneService: yes/no -> Yes/No
    df["PhoneService"] = df["PhoneService"].str.capitalize()

    # 4.2. tenure: "unknown" -> vazio
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")

    # 4.3. TotalCharges vazio com tenure = 0 -> 0 (cliente sem fatura fechada)
    sem_fatura = df["TotalCharges"].isna() & (df["tenure"] == 0)
    df.loc[sem_fatura, "TotalCharges"] = 0

    # 4.4. tenure vazio -> TotalCharges / MonthlyCharges (limitado a 0-72 meses)
    tenure_vazio = df["tenure"].isna()
    tenure_estimado = (df["TotalCharges"] / df["MonthlyCharges"]).round().clip(0, 72)
    df.loc[tenure_vazio, "tenure"] = tenure_estimado[tenure_vazio]

    # 5.1. n_adicionais: serviços de proteção e suporte contratados (0 a 4)
    df["n_adicionais"] = (
        (df["OnlineSecurity"] == "Yes").astype(int)
        + (df["OnlineBackup"] == "Yes").astype(int)
        + (df["DeviceProtection"] == "Yes").astype(int)
        + (df["TechSupport"] == "Yes").astype(int)
    )

    # 5.2. inicio_relacionamento: até 9 meses de casa
    df["inicio_relacionamento"] = np.where(df["tenure"] <= 9, "Yes", "No")
    df.loc[df["tenure"].isna(), "inicio_relacionamento"] = np.nan

    # 5.3. mensal_fibra: contrato mensal + fibra
    df["mensal_fibra"] = np.where(
        (df["Contract"] == "Month-to-month") & (df["InternetService"] == "Fiber optic"), "Yes", "No"
    )

    # 5.4. pagamento_automatico: transferência ou cartão automáticos
    df["pagamento_automatico"] = np.where(df["PaymentMethod"].str.contains("automatic"), "Yes", "No")

    # 5.5. razao_cobranca: TotalCharges / (tenure x MonthlyCharges)
    df["razao_cobranca"] = df["TotalCharges"] / (df["tenure"] * df["MonthlyCharges"])
    df["razao_cobranca"] = df["razao_cobranca"].replace([np.inf, -np.inf], np.nan)

    return df


# ==============================================================================================
# Modelo e carteira: carregados uma vez e guardados em cache
# ==============================================================================================
@st.cache_resource
def carregar_modelo():
    return joblib.load(ARQUIVO_MODELO)


@st.cache_data
def carregar_carteira():
    carteira = preparar_dados(pd.read_csv(ARQUIVO_CARTEIRA))
    artefato = carregar_modelo()
    probabilidade = artefato["modelo"].predict_proba(carteira[artefato["colunas"]])[:, 1]
    carteira["churn"] = np.where(probabilidade >= artefato["limiar"], "Sim", "Não")
    return carteira.set_index("customerID")


# ==============================================================================================
# Tela
# ==============================================================================================
st.set_page_config(page_title="Terra Signal · Churn", page_icon="📡", layout="centered")

st.title("📡 Esse cliente vai sair?")

carteira = carregar_carteira()

customer_id = st.selectbox(
    "Cliente",
    options=carteira.index.sort_values(),
    index=None,
    placeholder="Digite o customerID (ex.: 4226-KKDON)",
    label_visibility="collapsed",
)

if customer_id:
    cliente = carteira.loc[customer_id]

    if cliente["churn"] == "Sim":
        cor_fundo, cor_texto, titulo = "#FDECEA", "#B71C1C", "⚠️ SIM, risco de churn"
    else:
        cor_fundo, cor_texto, titulo = "#ECEFF1", "#37474F", "✅ NÃO, cliente estável"
    st.markdown(
        f"<div style='background:{cor_fundo};color:{cor_texto};padding:20px;border-radius:10px;"
        f"font-size:1.8rem;font-weight:700;text-align:center;'>{titulo}</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    coluna_1, coluna_2, coluna_3 = st.columns(3)
    coluna_1.metric("Mensalidade", f"R$ {cliente['MonthlyCharges']:.2f}".replace(".", ","))
    coluna_2.metric("Tempo de casa", f"{cliente['tenure']:.0f} meses")
    coluna_3.metric("Contrato", CONTRATO[cliente["Contract"]])

    # --- Serviços: o que o cliente já tem e o que ainda pode ser oferecido ---
    if cliente["PhoneService"] == "Yes":
        telefone = "2 ou mais linhas" if cliente["MultipleLines"] == "Yes" else "1 linha"
    else:
        telefone = "não tem"
    cartoes = ""
    for rotulo, valor in [("Internet", INTERNET[cliente["InternetService"]]),
                          ("Telefone", telefone),
                          ("Pagamento", PAGAMENTO[cliente["PaymentMethod"]])]:
        cartoes += (f"<div style='flex:1;background:#E3F2FD;border-radius:10px;padding:12px 14px;'>"
                    f"<div style='font-size:0.85rem;color:#1565C0;'>{rotulo}</div>"
                    f"<div style='font-size:1.25rem;font-weight:700;color:#0D47A1;'>{valor}</div></div>")
    st.markdown(f"<div style='display:flex;gap:10px;margin:8px 0 16px 0;'>{cartoes}</div>",
                unsafe_allow_html=True)

    # --- Adicionais: primeiro o que oferecer, depois o que o cliente já tem ---
    if cliente["InternetService"] == "No":
        st.caption("Sem internet: os serviços adicionais não estão disponíveis.")
    else:
        oferecer = [nome for coluna, nome in ADICIONAIS.items() if cliente[coluna] != "Yes"]
        ja_tem = [nome for coluna, nome in ADICIONAIS.items() if cliente[coluna] == "Yes"]

        if oferecer:
            selos = " ".join(f"<span style='background:#FFF3E0;color:#E65100;border:2px solid #FB8C00;"
                             f"padding:4px 12px;border-radius:14px;margin:3px;display:inline-block;"
                             f"font-weight:600;'>➕ {nome}</span>" for nome in oferecer)
            st.markdown(f"**🎯 Pode oferecer**<br>{selos}", unsafe_allow_html=True)
        else:
            st.markdown("**🎯 Pode oferecer:** o cliente já tem todos os adicionais.")

        if ja_tem:
            selos = " ".join(f"<span style='background:#F5F5F5;color:#757575;padding:2px 10px;"
                             f"border-radius:12px;margin:2px;display:inline-block;font-size:0.9rem;'>"
                             f"✓ {nome}</span>" for nome in ja_tem)
            st.markdown(f"<span style='color:#757575;font-size:0.9rem;'>Já tem</span><br>{selos}",
                        unsafe_allow_html=True)
