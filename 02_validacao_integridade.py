# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 02 — Validação de completude e integridade dos dados
# MAGIC
# MAGIC Execute este notebook após gerar `history_tratado` e `inference_tratado`,
# MAGIC antes de qualquer separação treino/teste. Ele **não altera** os dados:
# MAGIC apenas produz evidências de qualidade e interrompe o fluxo se encontrar
# MAGIC falhas críticas.

# COMMAND ----------

import numpy as np
import pandas as pd

CATALOG = "workspace.default"
TABELA_HISTORY = f"{CATALOG}.history_tratado"
TABELA_INFERENCE = f"{CATALOG}.inference_tratado"
TABELA_AUDITORIA = f"{CATALOG}.auditoria_integridade_modelagem"

history = spark.table(TABELA_HISTORY).toPandas()
inference = spark.table(TABELA_INFERENCE).toPandas()

print(f"history_tratado: {history.shape}")
print(f"inference_tratado: {inference.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Regras críticas de estrutura
# MAGIC
# MAGIC Estas regras precisam passar para que a base possa seguir para modelagem.

# COMMAND ----------

TARGET = "Churn"
ID_COL = "customerID"
# Usadas nas regras de integridade de domínio.
NUMERICAS_OBRIGATORIAS = ["tenure", "MonthlyCharges", "TotalCharges", "MonthlyIncome"]
# Usadas para comparar distribuição entre treino e inferência. A razão é uma
# feature contínua calculada, portanto precisa estar aqui, não nas categóricas.
NUMERICAS_ANALISE = [*NUMERICAS_OBRIGATORIAS, "fatura_sobre_renda"]
VALORES_AUSENTES_TEXTO = {"", " ", "unknown", "null", "none", "nan", "n/a", "na", "?"}

def texto_ausente(serie: pd.Series) -> pd.Series:
    """Reconhece NaN e os nulos escritos como texto."""
    return serie.isna() | serie.astype(str).str.strip().str.lower().isin(VALORES_AUSENTES_TEXTO)

history_tem_id = ID_COL in history.columns
inference_tem_id = ID_COL in inference.columns
ids_history = history[ID_COL] if history_tem_id else pd.Series(dtype="object")
ids_inference = inference[ID_COL] if inference_tem_id else pd.Series(dtype="object")

regras_criticas = {
    "history_tem_target": TARGET in history.columns,
    "inference_sem_target": TARGET not in inference.columns,
    "mesmas_features": set(history.columns) - {TARGET} == set(inference.columns),
    "id_presente_history": history_tem_id,
    "id_presente_inference": inference_tem_id,
    "id_unico_history": history_tem_id and ids_history.notna().all() and not ids_history.duplicated().any(),
    "id_unico_inference": inference_tem_id and ids_inference.notna().all() and not ids_inference.duplicated().any(),
    "sem_id_repetido_entre_bases": history_tem_id and inference_tem_id and not set(ids_history).intersection(set(ids_inference)),
    "target_valido": TARGET in history.columns and set(history[TARGET].dropna().unique()).issubset({"Yes", "No", 0, 1}),
}

resultado_regras = pd.DataFrame(
    [{"regra": regra, "status": "PASSOU" if passou else "FALHOU"}
     for regra, passou in regras_criticas.items()]
)
display(resultado_regras)

falhas_criticas = resultado_regras.query("status == 'FALHOU'")
assert falhas_criticas.empty, (
    "Falhas críticas encontradas. Corrija a etapa de tratamento antes de modelar:\n"
    + ", ".join(falhas_criticas["regra"])
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Completude: nulos reais e nulos escritos como texto
# MAGIC
# MAGIC Depois de tratada, a base de treino não deve conter `unknown`, `null`,
# MAGIC strings vazias ou `NaN` em nenhuma feature usada no modelo. Flags como
# MAGIC `tenure_missing` e `tc_missing` podem permanecer: elas representam o fato
# MAGIC de que o valor original esteve ausente.

# COMMAND ----------

def relatorio_completude(df: pd.DataFrame, nome_base: str) -> pd.DataFrame:
    linhas = []
    for coluna in df.columns:
        ausente = texto_ausente(df[coluna])
        linhas.append({
            "base": nome_base,
            "coluna": coluna,
            "linhas_ausentes": int(ausente.sum()),
            "pct_ausente": round(100 * ausente.mean(), 3),
            "exemplos": df.loc[ausente, ID_COL].head(5).astype(str).tolist(),
        })
    return pd.DataFrame(linhas)

completude = pd.concat([
    relatorio_completude(history, "history_tratado"),
    relatorio_completude(inference, "inference_tratado"),
], ignore_index=True)

display(completude.sort_values(["linhas_ausentes", "base"], ascending=[False, True]))

pendencias_completude = completude.query("linhas_ausentes > 0")
if pendencias_completude.empty:
    print("PASSOU: não há nulos ou valores textuais ausentes nas tabelas tratadas.")
else:
    print("ATENÇÃO: há pendências de completude. Não avance sem decidir como tratá-las.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Tipos, domínios e regras de negócio
# MAGIC
# MAGIC Alguns desvios são alertas para investigação, não correções automáticas.
# MAGIC Por exemplo, `TotalCharges` pode não ser exatamente `tenure × MonthlyCharges`
# MAGIC por descontos, mudanças de plano ou cobranças proporcionais.

# COMMAND ----------

def relatorio_integridade(df: pd.DataFrame, nome_base: str) -> pd.DataFrame:
    teste = df.copy()
    for coluna in NUMERICAS_OBRIGATORIAS:
        teste[coluna] = pd.to_numeric(teste[coluna], errors="coerce")

    razao = teste["TotalCharges"] / (teste["tenure"] * teste["MonthlyCharges"])
    regras = {
        "numericas_convertiveis": teste[NUMERICAS_OBRIGATORIAS].notna().all(axis=1),
        "tenure_nao_negativo": teste["tenure"].ge(0),
        "monthly_charges_positivo": teste["MonthlyCharges"].gt(0),
        "total_charges_nao_negativo": teste["TotalCharges"].ge(0),
        "monthly_income_positivo": teste["MonthlyIncome"].gt(0),
        "phone_service_padronizado": teste["PhoneService"].isin(["Yes", "No"]),
        "sem_internet_tem_servico_coerente": ~(
            teste["InternetService"].eq("No")
            & teste["OnlineSecurity"].ne("No internet service")
        ),
        "total_charges_razoavel": (
            teste["tenure"].eq(0)
            | razao.between(0.70, 1.30)
        ),
    }

    linhas = []
    for nome_regra, passou in regras.items():
        falhou = ~passou.fillna(False)
        linhas.append({
            "base": nome_base,
            "regra": nome_regra,
            "falhas": int(falhou.sum()),
            "pct_falhas": round(100 * falhou.mean(), 3),
            "exemplos_id": teste.loc[falhou, ID_COL].head(5).astype(str).tolist(),
        })
    return pd.DataFrame(linhas)

integridade = pd.concat([
    relatorio_integridade(history, "history_tratado"),
    relatorio_integridade(inference, "inference_tratado"),
], ignore_index=True)
display(integridade.sort_values(["falhas", "base"], ascending=[False, True]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Comparabilidade entre treino e inferência
# MAGIC
# MAGIC O objetivo é detectar categorias presentes apenas em uma das bases e
# MAGIC grandes diferenças de distribuição. Não é um teste de causalidade nem uma
# MAGIC justificativa para alterar a inferência; é uma evidência de risco de drift.

# COMMAND ----------

numericas_analise_presentes = [c for c in NUMERICAS_ANALISE if c in inference.columns]
categoricas = [
    c for c in inference.columns
    if c not in numericas_analise_presentes + [ID_COL, "CustomerFeedback"]
]

linhas_categorias = []
for coluna in categoricas:
    categorias_history = set(history[coluna].dropna().astype(str))
    categorias_inference = set(inference[coluna].dropna().astype(str))
    linhas_categorias.append({
        "coluna": coluna,
        "so_history": sorted(categorias_history - categorias_inference),
        "so_inference": sorted(categorias_inference - categorias_history),
    })

categorias_novas = pd.DataFrame(linhas_categorias)
display(categorias_novas)

comparativo_numerico = pd.DataFrame({
    "media_history": history[numericas_analise_presentes].apply(pd.to_numeric, errors="coerce").mean(),
    "media_inference": inference[numericas_analise_presentes].apply(pd.to_numeric, errors="coerce").mean(),
    "mediana_history": history[numericas_analise_presentes].apply(pd.to_numeric, errors="coerce").median(),
    "mediana_inference": inference[numericas_analise_presentes].apply(pd.to_numeric, errors="coerce").median(),
})
comparativo_numerico["dif_media_pct"] = (
    (comparativo_numerico["media_inference"] - comparativo_numerico["media_history"])
    / comparativo_numerico["media_history"].replace(0, np.nan)
)
display(comparativo_numerico.round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Evidência da auditoria
# MAGIC
# MAGIC A tabela de auditoria guarda os resultados para consulta e para os slides.
# MAGIC Os dados de treino e inferência permanecem intactos.

# COMMAND ----------

def serializar_lista(valor):
    if isinstance(valor, (list, tuple, set, np.ndarray)):
        return ", ".join(map(str, valor))
    if pd.isna(valor):
        return ""
    return str(valor)


TABELA_ESTRUTURA = "workspace.default.auditoria_estrutura_modelagem"
TABELA_COMPLETUDE = "workspace.default.auditoria_completude_modelagem"
TABELA_INTEGRIDADE = "workspace.default.auditoria_integridade_modelagem"


# 1. Estrutura: uma linha por regra crítica
auditoria_estrutura = resultado_regras.copy()


# 2. Completude: uma linha por coluna e por base
auditoria_completude = completude.rename(columns={
    "linhas_ausentes": "qtd_problemas",
    "pct_ausente": "pct_problemas",
}).copy()

auditoria_completude["status"] = np.where(
    auditoria_completude["qtd_problemas"].eq(0),
    "PASSOU",
    "ALERTA"
)

auditoria_completude["exemplos"] = (
    auditoria_completude["exemplos"].apply(serializar_lista)
)

auditoria_completude = auditoria_completude[[
    "base",
    "coluna",
    "status",
    "qtd_problemas",
    "pct_problemas",
    "exemplos"
]]


# 3. Integridade: uma linha por regra de negócio e por base
auditoria_integridade = integridade.rename(columns={
    "falhas": "qtd_problemas",
    "pct_falhas": "pct_problemas",
    "exemplos_id": "exemplos",
}).copy()

auditoria_integridade["status"] = np.where(
    auditoria_integridade["qtd_problemas"].eq(0),
    "PASSOU",
    "ALERTA"
)

auditoria_integridade["exemplos"] = (
    auditoria_integridade["exemplos"].apply(serializar_lista)
)

auditoria_integridade = auditoria_integridade[[
    "base",
    "regra",
    "status",
    "qtd_problemas",
    "pct_problemas",
    "exemplos"
]]


# Salvar as três evidências
(spark.createDataFrame(auditoria_estrutura)
      .write.mode("overwrite")
      .option("overwriteSchema", "true")
      .saveAsTable(TABELA_ESTRUTURA))

(spark.createDataFrame(auditoria_completude)
      .write.mode("overwrite")
      .option("overwriteSchema", "true")
      .saveAsTable(TABELA_COMPLETUDE))

(spark.createDataFrame(auditoria_integridade)
      .write.mode("overwrite")
      .option("overwriteSchema", "true")
      .saveAsTable(TABELA_INTEGRIDADE))


display(auditoria_estrutura)
display(auditoria_completude)
display(auditoria_integridade)

print("Tabelas de auditoria salvas:")
print(f"- {TABELA_ESTRUTURA}")
print(f"- {TABELA_COMPLETUDE}")
print(f"- {TABELA_INTEGRIDADE}")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * 
# MAGIC FROM workspace.default.auditoria_integridade_modelagem

# COMMAND ----------

