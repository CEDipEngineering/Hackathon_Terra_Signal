"""App de predição de churn de clientes.

Recebe os dados de um cliente via formulário e consulta o endpoint de
Model Serving `churn-prediction-endpoint`, que serve o modelo registrado
no Unity Catalog (workspace.default.hackathon_databricks_model) e retorna
a probabilidade de churn do cliente.
"""

import requests
import streamlit as st
from databricks.sdk.core import Config

ENDPOINT = "churn-prediction-endpoint"

st.set_page_config(page_title="Predição de Churn", page_icon="📉", layout="wide")

st.title("📉 Predição de Churn")
st.caption(
    "Preencha os dados do cliente para estimar a probabilidade de cancelamento "
    "do serviço, com base no modelo registrado em "
    "`workspace.default.hackathon_databricks_model` (servido via endpoint "
    "de Model Serving)."
)

st.sidebar.caption(f"Scoring via endpoint `{ENDPOINT}`")


def predict_churn(record: dict) -> float:
    """Consulta o endpoint de serving e retorna a probabilidade de churn."""
    cfg = Config()  # credenciais OAuth do service principal do app
    resp = requests.post(
        f"{cfg.host.rstrip('/')}/serving-endpoints/{ENDPOINT}/invocations",
        headers=cfg.authenticate(),
        json={"dataframe_records": [record]},
        timeout=(10, 300),  # cold start pode levar alguns minutos
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
    return float(resp.json()["predictions"][0])


with st.form("dados_cliente", border=True):
    st.subheader("Dados do cliente")

    c1, c2 = st.columns(2)
    gender = c1.selectbox("Gênero", ["Male", "Female"])
    senior = c2.selectbox("Idoso (65+)", ["Não", "Sim"])
    senior_citizen = 1 if senior == "Sim" else 0

    c4, c5, c6 = st.columns(3)
    partner = c4.selectbox("Possui parceiro(a)?", ["No", "Yes"])
    dependents = c5.selectbox("Possui dependentes?", ["No", "Yes"])
    tenure = c6.number_input(
        "Meses de contrato (tenure)", min_value=0, max_value=120, value=12, step=1
    )

    st.subheader("Serviços contratados")

    c7, c8 = st.columns(2)
    phone_service = c7.selectbox("Serviço de telefone", ["Yes", "No"])
    multiple_lines = c8.selectbox(
        "Múltiplas linhas", ["No", "Yes", "No phone service"]
    )
    internet_service = c7.selectbox(
        "Serviço de internet", ["DSL", "Fiber optic", "No"]
    )

    no_internet = "No internet service"
    g1, g2, g3 = st.columns(3)
    online_security = g1.selectbox("Segurança online", ["Yes", "No", no_internet])
    online_backup = g2.selectbox("Backup online", ["No", "Yes", no_internet])
    device_protection = g3.selectbox("Proteção de dispositivo", ["No", "Yes", no_internet])

    g4, g5, g6 = st.columns(3)
    tech_support = g4.selectbox("Suporte técnico", ["No", "Yes", no_internet])
    streaming_tv = g5.selectbox("Streaming TV", ["No", "Yes", no_internet])
    streaming_movies = g6.selectbox("Streaming filmes", ["No", "Yes", no_internet])

    st.subheader("Contrato e cobrança")

    h1, h2 = st.columns(2)
    contract = h1.selectbox(
        "Tipo de contrato", ["Month-to-month", "One year", "Two year"]
    )
    paperless_billing = h2.selectbox("Fatura digital (paperless)", ["Yes", "No"])
    payment_method = h1.selectbox(
        "Forma de pagamento",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    i1, i2 = st.columns(2)
    monthly_charges = i1.number_input(
        "Cobrança mensal (R$/mês)", min_value=0.0, value=70.0, step=0.1
    )
    total_charges = i2.number_input(
        "Cobrança total acumulada", min_value=0.0, value=500.0, step=0.1
    )

    st.subheader("Perfil socioeconômico")

    monthly_income = st.number_input(
        "Renda mensal (R$)", min_value=0, value=5000, step=100
    )

    submitted = st.form_submit_button("🔮 Calcular probabilidade de churn",
                                     use_container_width=True)

if submitted:
    registro = {
        "customerID": "não informado",
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": str(int(tenure)),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
        "CustomerFeedback": "não informado",
        "MonthlyIncome": int(monthly_income),
        "motivo_feedback": "não informado",
    }

    try:
        with st.spinner(
            "Consultando o endpoint de serving... (a primeira chamada após "
            "um período de inatividade pode levar alguns minutos)"
        ):
            churn_prob = predict_churn(registro)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Falha ao calcular a predição: {exc}")
        st.stop()

    st.divider()
    st.subheader("Resultado")

    if churn_prob >= 0.7:
        nivel, emoji = "Alto risco de churn", "🔴"
    elif churn_prob >= 0.3:
        nivel, emoji = "Risco moderado", "🟠"
    else:
        nivel, emoji = "Baixo risco", "🟢"

    k1, k2 = st.columns(2)
    k1.metric("Probabilidade de churn", f"{churn_prob:.1%}")
    k2.metric("Classificação", f"{emoji} {nivel}")

    st.progress(min(churn_prob, 1.0), text=f"Probabilidade de churn: {churn_prob:.1%}")

    with st.expander("Ver dados enviados ao modelo"):
        st.json(registro)