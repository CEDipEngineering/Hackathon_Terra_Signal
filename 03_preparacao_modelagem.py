# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 03 — Preparação e triagem de features para modelagem de churn
# MAGIC
# MAGIC Este notebook cria uma camada **model-ready**, mas não transforma
# MAGIC categorias em dummies. Essa transformação deve ocorrer dentro do pipeline
# MAGIC de treino, para que a cross-validation não receba informação do conjunto
# MAGIC de validação.

# COMMAND ----------

import numpy as np
import pandas as pd

CATALOG = "workspace.default"
TABELA_HISTORY = f"{CATALOG}.history_tratado"
TABELA_INFERENCE = f"{CATALOG}.inference_tratado"
SAIDA_HISTORY = f"{CATALOG}.history_modelagem"
SAIDA_INFERENCE = f"{CATALOG}.inference_modelagem"

history = spark.table(TABELA_HISTORY).toPandas()
inference = spark.table(TABELA_INFERENCE).toPandas()

print(f"history: {history.shape} | inference: {inference.shape}")
assert "Churn" in history.columns
assert "Churn" not in inference.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Features candidatas e decisões de escopo
# MAGIC
# MAGIC `CustomerFeedback` e `declarou_saida` ficam fora do modelo principal:
# MAGIC ambos contêm sinais de intenção de cancelamento e podem causar leakage.
# MAGIC `customerID` é apenas identificador. `gender` e `SeniorCitizen` ficam na
# MAGIC análise de viés; a decisão de usá-los no modelo deve ser comparada depois.
# MAGIC
# MAGIC `PaymentMethod` já contém a categoria `Electronic check`. Por isso, as
# MAGIC flags derivadas de pagamento abaixo são úteis para EDA e para um cenário
# MAGIC compacto, mas não devem entrar junto com `PaymentMethod` no mesmo modelo.

# COMMAND ----------

ID_COL = "customerID"
TARGET = "Churn"
COLUNAS_LEAKAGE = ["CustomerFeedback", "declarou_saida"]
ATRIBUTOS_SENSIVEIS = ["gender", "SeniorCitizen"]

SERVICOS_ADICIONAIS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"
]
FORMAS_AUTOMATICAS = ["Credit card (automatic)", "Bank transfer (automatic)"]

def criar_features(df_original: pd.DataFrame, tem_target: bool) -> pd.DataFrame:
    """Mantém variáveis de negócio e cria poucas features determinísticas."""
    df = df_original.copy()

    # A preparação anterior já corrigiu os tipos; estas conversões tornam o
    # notebook seguro caso uma tabela seja recriada a partir do CSV.
    for coluna in ["tenure", "MonthlyCharges", "TotalCharges", "MonthlyIncome"]:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    # Quantidade de serviços de retenção: 0 a 4. "No internet service" não conta.
    df["qtd_servicos_adicionais"] = (
        df[SERVICOS_ADICIONAIS].eq("Yes").sum(axis=1).astype("int64")
    )
    df["tem_internet"] = np.where(df["InternetService"].eq("No"), "No", "Yes")

    # A flag pedida pela equipe: "Yes" significa especificamente Electronic check.
    # Mantemos também a forma automática, pois ela representa uma hipótese de negócio
    # diferente. Ambas são derivadas de PaymentMethod e, portanto, redundantes com ele.
    df["usa_electronic_check"] = np.where(
        df["PaymentMethod"].eq("Electronic check"), "Yes", "No"
    )
    df["pagamento_automatico"] = np.where(
        df["PaymentMethod"].isin(FORMAS_AUTOMATICAS), "Yes", "No"
    )

    # Razão é mantida por ser interpretável: quanto a fatura pesa na renda.
    # Valores impossíveis ou renda zero viram ausentes para imputação no pipeline.
    df["fatura_sobre_renda"] = df["MonthlyCharges"] / df["MonthlyIncome"].replace(0, np.nan)
    df.loc[~np.isfinite(df["fatura_sobre_renda"]), "fatura_sobre_renda"] = np.nan

    if tem_target:
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
        if df[TARGET].isna().any():
            raise ValueError("Churn precisa conter somente Yes ou No.")

    return df

history_feat = criar_features(history, tem_target=True)
inference_feat = criar_features(inference, tem_target=False)

COLUNAS_EXCLUIDAS = [ID_COL, *COLUNAS_LEAKAGE]
features_candidatas = [
    c for c in inference_feat.columns if c not in COLUNAS_EXCLUIDAS
]

# A lista abaixo é a recomendada para o primeiro modelo de negócio.
# Atributos sensíveis continuam disponíveis apenas para auditoria e comparação.
features_principais = [c for c in features_candidatas if c not in ATRIBUTOS_SENSIVEIS]

# Escolher uma única representação de pagamento no notebook de modelagem:
# - cenário completo: usar PaymentMethod e excluir as duas flags;
# - cenário compacto: usar usa_electronic_check e pagamento_automatico e excluir PaymentMethod.
COLUNAS_PAGAMENTO_DERIVADAS = ["usa_electronic_check", "pagamento_automatico"]
features_pagamento_completo = [
    c for c in features_principais if c not in COLUNAS_PAGAMENTO_DERIVADAS
]
features_pagamento_compacto = [
    c for c in features_principais if c != "PaymentMethod"
]

print("Features candidatas:", len(features_candidatas))
print("Features principais (sem atributos sensíveis):", len(features_principais))
print("Excluídas por ID/leakage:", COLUNAS_EXCLUIDAS)
print("Cenário com PaymentMethod:", len(features_pagamento_completo), "features")
print("Cenário com flags de pagamento:", len(features_pagamento_compacto), "features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Triagem antes do encoding
# MAGIC
# MAGIC Esta análise acontece com as categorias legíveis (`Contract`,
# MAGIC `PaymentMethod`, etc.). Ela serve para entender o negócio e identificar
# MAGIC variáveis constantes, muito ausentes ou redundantes. Não use apenas esta
# MAGIC triagem para remover uma feature: a confirmação vem da cross-validation.

# COMMAND ----------

numericas = history_feat[features_candidatas].select_dtypes(include=np.number).columns.tolist()
categoricas = [c for c in features_candidatas if c not in numericas]

qualidade = pd.DataFrame({
    "feature": features_candidatas,
    "tipo": ["numérica" if c in numericas else "categórica" for c in features_candidatas],
    "n_unicos": [history_feat[c].nunique(dropna=True) for c in features_candidatas],
    "pct_ausente": [history_feat[c].isna().mean() for c in features_candidatas],
})

display(qualidade.sort_values(["pct_ausente", "n_unicos"], ascending=[False, True]))

if numericas:
    corr_numericas = history_feat[numericas + [TARGET]].corr(method="spearman")[TARGET].drop(TARGET)
    display(
        corr_numericas.rename("spearman_com_churn")
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .to_frame()
    )

    # Correlação alta entre atributos não implica remover automaticamente uma coluna.
    # Ela sinaliza atenção na interpretação dos coeficientes da regressão logística.
    display(history_feat[numericas].corr(method="spearman").round(2))

# COMMAND ----------

def resumo_categorica(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """Taxa e lift de churn por categoria; não estabelece causalidade."""
    geral = df[TARGET].mean()
    tabela = (
        df.groupby(coluna, dropna=False)[TARGET]
        .agg(clientes="size", churners="sum", taxa_churn="mean")
        .reset_index()
    )
    tabela["lift_vs_base"] = tabela["taxa_churn"] / geral
    return tabela.sort_values("taxa_churn", ascending=False)

for coluna in categoricas:
    print(f"\n--- {coluna} ---")
    display(resumo_categorica(history_feat, coluna))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Checagens específicas para seleção de features
# MAGIC
# MAGIC - Uma feature com grupos 50/50 pode ser preditiva; compare a taxa de churn,
# MAGIC   não apenas a distribuição.
# MAGIC - `TotalCharges` é próximo de `tenure × MonthlyCharges`; avalie a redundância
# MAGIC   nos coeficientes da logística. O primeiro experimento pode usar os três e
# MAGIC   um segundo pode remover `TotalCharges`.
# MAGIC - Features são mantidas nesta tabela. A seleção final ocorre somente dentro
# MAGIC   da cross-validation do notebook de modelagem.

# COMMAND ----------

for coluna in ATRIBUTOS_SENSIVEIS:
    print(f"\nDiagnóstico de possível viés — {coluna}")
    display(resumo_categorica(history_feat, coluna))

print("\nDistribuição do alvo:")
display(history_feat[TARGET].value_counts(normalize=True).rename("proporcao").to_frame())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Persistir a camada model-ready
# MAGIC
# MAGIC Não há dummies, normalização ou imputação por mediana aqui. Elas serão
# MAGIC ajustadas pelo `Pipeline` sobre cada fold de treino no próximo notebook.

# COMMAND ----------

colunas_saida_history = [TARGET] + features_candidatas
colunas_saida_inference = features_candidatas

history_modelagem = history_feat[colunas_saida_history].copy()
inference_modelagem = inference_feat[colunas_saida_inference].copy()

assert list(history_modelagem.drop(columns=TARGET).columns) == list(inference_modelagem.columns)
assert not set(COLUNAS_EXCLUIDAS).intersection(history_modelagem.columns)

(spark.createDataFrame(history_modelagem)
      .write.mode("overwrite")
      .option("overwriteSchema", "true")
      .saveAsTable(SAIDA_HISTORY))

(spark.createDataFrame(inference_modelagem)
      .write.mode("overwrite")
      .option("overwriteSchema", "true")
      .saveAsTable(SAIDA_INFERENCE))

print(f"Tabelas criadas: {SAIDA_HISTORY} e {SAIDA_INFERENCE}")
print("As bases de dados estão prontas para treino/teste estratificado e ajuste de Pipeline.")

# COMMAND ----------

