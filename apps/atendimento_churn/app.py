"""App simples de apoio ao atendimento preventivo da Terra Signal."""

import os

import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config

TABELA_CLIENTES = "workspace.default.clientes_risco_churn"
TABELA_ATENDIMENTOS = "workspace.default.atendimentos_churn"

st.set_page_config(page_title="Terra Signal | Atendimento", page_icon="☎️", layout="wide")


@st.cache_resource
def conexao_sql():
    """Cria uma conexão autenticada pela identidade de serviço do app."""
    warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
    if not warehouse_id:
        raise RuntimeError(
            "O recurso SQL Warehouse não foi configurado no app. "
            "Defina DATABRICKS_WAREHOUSE_ID no app.yaml."
        )

    cfg = Config()
    hostname = cfg.host.replace("https://", "").replace("http://", "")
    return sql.connect(
        server_hostname=hostname,
        http_path=f"/sql/1.0/warehouses/{warehouse_id}",
        credentials_provider=lambda: cfg.authenticate,
    )


def consultar_cliente(customer_id: str) -> pd.Series | None:
    consulta = f"""
        SELECT customerID, tenure, Contract, PaymentMethod, InternetService,
               TechSupport, MonthlyCharges, score_churn, no_top_10, no_top_20,
               no_top_30, top_faixa, faixa_risco, segmento_acao, acao_sugerida
        FROM {TABELA_CLIENTES}
        WHERE customerID = ?
        LIMIT 1
    """
    with conexao_sql().cursor() as cursor:
        cursor.execute(consulta, [customer_id.strip()])
        dados = cursor.fetchall_arrow().to_pandas()
    return None if dados.empty else dados.iloc[0]


def para_bool(valor) -> bool:
    return str(valor).strip().lower() in {"true", "1", "yes", "sim"}


def salvar_atendimento(cliente: pd.Series, dados_formulario: dict) -> None:
    comando = f"""
        INSERT INTO {TABELA_ATENDIMENTOS} (
            data_atendimento, customerID, score_churn, top_faixa, faixa_risco,
            segmento_acao, acao_sugerida, motivo_atendimento,
            recebeu_acao_promocional_recente, desfecho_chamado,
            avaliacao_servico, avaliacao_atendimento, sugestao_cliente
        ) VALUES (current_timestamp(), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    parametros = [
        str(cliente["customerID"]), float(cliente["score_churn"]),
        str(cliente["top_faixa"]), str(cliente["faixa_risco"]),
        str(cliente["segmento_acao"]), str(cliente["acao_sugerida"]),
        dados_formulario["motivo_atendimento"],
        dados_formulario["acao_promocional_recente"],
        dados_formulario["desfecho_chamado"],
        dados_formulario["avaliacao_servico"],
        dados_formulario["avaliacao_atendimento"],
        dados_formulario["sugestao_cliente"],
    ]
    with conexao_sql().cursor() as cursor:
        cursor.execute(comando, parametros)


st.title("Atendimento preventivo de churn")
st.caption(
    "Consulte o contexto de risco e registre o atendimento. "
    "O score prioriza risco e não representa uma certeza de cancelamento."
)

with st.form("busca_cliente"):
    customer_id = st.text_input(
        "Customer ID", placeholder="Ex.: 0107-YHINA",
        help="Informe o identificador do cliente presente na carteira de risco.",
    )
    buscar = st.form_submit_button("Buscar cliente", type="primary")

if buscar:
    st.session_state.pop("cliente_encontrado", None)
    if not customer_id.strip():
        st.warning("Informe um customerID para consultar a carteira.")
    else:
        try:
            cliente = consultar_cliente(customer_id)
        except Exception as erro:
            st.error("Não foi possível consultar a carteira de risco.")
            st.exception(erro)
            cliente = None
        if cliente is None:
            st.info("Cliente não encontrado na carteira de scoring.")
        else:
            st.session_state["cliente_encontrado"] = cliente.to_dict()

cliente_dict = st.session_state.get("cliente_encontrado")
if cliente_dict:
    cliente = pd.Series(cliente_dict)
    st.divider()
    st.subheader(f"Cliente {cliente['customerID']}")
    col_score, col_faixa, col_segmento = st.columns(3)
    col_score.metric("Score de risco de churn", f"{float(cliente['score_churn']):.1%}")
    col_faixa.metric("Faixa de priorização", cliente["top_faixa"])
    col_segmento.metric("Segmento de ação", cliente["segmento_acao"])
    st.info(
        "Use este score para priorizar o atendimento. Ele não prova causalidade "
        "nem é uma probabilidade perfeitamente calibrada de churn."
    )

    st.subheader("Perfil relevante para o atendimento")
    perfil_1, perfil_2, perfil_3 = st.columns(3)
    perfil_1.write(f"**Internet:** {cliente['InternetService']}")
    perfil_1.write(f"**Tenure:** {int(cliente['tenure'])} meses")
    perfil_2.write(f"**Contrato:** {cliente['Contract']}")
    perfil_2.write(f"**Pagamento:** {cliente['PaymentMethod']}")
    perfil_3.write(f"**Suporte técnico:** {cliente['TechSupport']}")
    perfil_3.write(f"**Cobrança mensal:** R$ {float(cliente['MonthlyCharges']):.2f}")

    st.subheader("Elegibilidade para ação preventiva")
    top_10, top_20, top_30 = st.columns(3)
    top_10.metric("Top 10%", "Selecionado" if para_bool(cliente["no_top_10"]) else "Não selecionado")
    top_20.metric("Top 20%", "Selecionado" if para_bool(cliente["no_top_20"]) else "Não selecionado")
    top_30.metric("Top 30%", "Selecionado" if para_bool(cliente["no_top_30"]) else "Não selecionado")
    st.caption("Os públicos são cumulativos: Top 20% inclui Top 10%; Top 30% inclui ambos.")
    st.subheader("Ação sugerida")
    st.success(cliente["acao_sugerida"])

    st.divider()
    st.subheader("Registro do atendimento")
    with st.form("registro_atendimento", clear_on_submit=True):
        motivo = st.text_area("Motivo do atendimento ou queixa", height=100)
        acao_recente = st.selectbox("Recebeu alguma ação promocional recente?", ["Não informado", "Não", "Sim"])
        desfecho = st.text_area("Desfecho do chamado", height=100)
        avaliacao_servico = st.text_area("Avaliação do serviço", height=80)
        avaliacao_atendimento = st.text_area("Avaliação do atendimento", height=80)
        sugestao = st.text_area("Sugestão do cliente", height=80)
        salvar = st.form_submit_button("Salvar registro", type="primary")

    if salvar:
        if not motivo.strip() or not desfecho.strip():
            st.warning("Preencha ao menos o motivo do atendimento e o desfecho do chamado.")
        else:
            try:
                salvar_atendimento(cliente, {
                    "motivo_atendimento": motivo.strip(),
                    "acao_promocional_recente": acao_recente,
                    "desfecho_chamado": desfecho.strip(),
                    "avaliacao_servico": avaliacao_servico.strip(),
                    "avaliacao_atendimento": avaliacao_atendimento.strip(),
                    "sugestao_cliente": sugestao.strip(),
                })
                st.success("Registro salvo com sucesso.")
            except Exception as erro:
                st.error("Não foi possível salvar o registro de atendimento.")
                st.exception(erro)
