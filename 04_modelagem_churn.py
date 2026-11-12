# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "shap",
# ]
# ///
# MAGIC %md
# MAGIC # 04 — Modelagem de churn e comparação de modelos
# MAGIC
# MAGIC Este notebook usa a camada `*_modelagem`, mantém um conjunto de teste
# MAGIC intocado e compara Regressão Logística, Árvore de Decisão e Random Forest.
# MAGIC A seleção inicial usa PR-AUC na cross-validation; recall, precision e F2
# MAGIC ajudam a avaliar o equilíbrio operacional da campanha de retenção.

# COMMAND ----------

import datetime
import os
import tempfile

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer

CATALOG = "workspace.default"
TABELA_HISTORY = f"{CATALOG}.history_modelagem"
TABELA_INFERENCE = f"{CATALOG}.inference_modelagem"
TABELA_INFERENCE_TRATADO = f"{CATALOG}.inference_tratado"
EXPERIMENTO = "/Shared/terra_signal_churn"
NOME_EQUIPE = "Equipe 5"
SEED = 42

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Carregar dados e definir o conjunto de features
# MAGIC
# MAGIC O cenário inicial mantém as quatro categorias de `PaymentMethod`. Portanto,
# MAGIC remove as duas flags derivadas de pagamento. `gender` e `SeniorCitizen`
# MAGIC também ficam fora do primeiro modelo, mas podem ser usados depois para
# MAGIC auditoria de desempenho por grupo.

# COMMAND ----------

history = spark.table(TABELA_HISTORY).toPandas()
inference = spark.table(TABELA_INFERENCE).toPandas()
ids_inference = spark.table(TABELA_INFERENCE_TRATADO).select("customerID").toPandas()

TARGET = "Churn"
ATRIBUTOS_SENSIVEIS = ["gender", "SeniorCitizen"]
FLAGS_PAGAMENTO = ["usa_electronic_check", "pagamento_automatico"]

features_modelo = [
    coluna for coluna in inference.columns
    if coluna not in ATRIBUTOS_SENSIVEIS + FLAGS_PAGAMENTO
]

assert TARGET in history.columns
assert set(features_modelo).issubset(history.columns)
assert list(inference.columns) == list(history.drop(columns=TARGET).columns)
assert len(ids_inference) == len(inference)

X = history[features_modelo].copy()
y = history[TARGET].astype(int).copy()
X_inference = inference[features_modelo].copy()

print(f"Treino histórico: {X.shape} | Inferência: {X_inference.shape}")
print(f"Taxa de churn: {y.mean():.1%}")
print("Features do primeiro cenário:")
print(features_modelo)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Separar treino e teste antes de qualquer ajuste
# MAGIC
# MAGIC A estratificação preserva a proporção de churn em ambos os conjuntos.
# MAGIC O teste fica reservado para a avaliação final e não participa da escolha
# MAGIC de hiperparâmetros ou da comparação por cross-validation.

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEED,
)

print(f"Treino: {X_train.shape[0]:,} linhas | churn: {y_train.mean():.1%}")
print(f"Teste:  {X_test.shape[0]:,} linhas | churn: {y_test.mean():.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Pipelines de preparação
# MAGIC
# MAGIC As imputações, dummies e normalização são ajustadas dentro de cada fold da
# MAGIC cross-validation. Assim, nenhum dado do fold de validação ou do teste vaza
# MAGIC para o treino. A normalização é aplicada apenas à regressão logística.

# COMMAND ----------

numericas = X_train.select_dtypes(include=np.number).columns.tolist()
categoricas = [coluna for coluna in X_train.columns if coluna not in numericas]

preprocessador_logistica = ColumnTransformer(
    transformers=[
        ("numericas", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numericas),
        ("categoricas", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categoricas),
    ],
    remainder="drop",
)

preprocessador_arvores = ColumnTransformer(
    transformers=[
        ("numericas", SimpleImputer(strategy="median"), numericas),
        ("categoricas", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categoricas),
    ],
    remainder="drop",
)

modelos = {
    "regressao_logistica": Pipeline([
        ("preprocessamento", preprocessador_logistica),
        ("modelo", LogisticRegression(
            max_iter=2_000,
            class_weight="balanced", 
            random_state=SEED
            )),
    ]),

    "arvore_decisao": Pipeline([
        ("preprocessamento", preprocessador_arvores),
        ("modelo", DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=30,
            class_weight="balanced",
            random_state=SEED,
        )),
    ]),
    "random_forest": Pipeline([
        ("preprocessamento", preprocessador_arvores),
        ("modelo", RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )),
    ]),
}

print("Numéricas:", numericas)
print("Categóricas:", categoricas)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Cross-validation no treino - para estabelecer baselines em Regressão Logística, Árvore de Decisão e Randon Forest
# MAGIC
# MAGIC PR-AUC é a métrica principal de comparação, pois a classe churn é minoritária.
# MAGIC F2 dá mais peso ao recall: perder um churner é mais caro do que abordar um
# MAGIC cliente que acabaria ficando. A decisão operacional de threshold virá depois.

# COMMAND ----------

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "f2": make_scorer(fbeta_score, beta=2),
}

resultados_cv = []

for nome, pipeline in modelos.items():
    resultado = cross_validate(
        pipeline,
        X_train,
        y_train,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )
    linha = {"modelo": nome}
    for metrica in scoring:
        linha[f"{metrica}_media"] = resultado[f"test_{metrica}"].mean()
        linha[f"{metrica}_desvio"] = resultado[f"test_{metrica}"].std()
    resultados_cv.append(linha)

resultados_cv = (
    pd.DataFrame(resultados_cv)
    .sort_values("pr_auc_media", ascending=False)
    .reset_index(drop=True)
)
display(resultados_cv.style.format({
    coluna: "{:.3f}" for coluna in resultados_cv.columns if coluna != "modelo"
}))

# COMMAND ----------

import mlflow

EXPERIMENTO = "/Shared/terra_signal_churn"

mlflow.set_experiment(EXPERIMENTO)

for _, resultado in resultados_cv.iterrows():
    nome_modelo = resultado["modelo"]
    modelo = modelos[nome_modelo].named_steps["modelo"]

    with mlflow.start_run(run_name=f"baseline__{nome_modelo}"):

        mlflow.set_tags({
            "etapa": "baseline_cv",
            "modelo": nome_modelo,
            "conjunto_features": "completo_sem_sensiveis",
            "pagamento": "PaymentMethod_completo",
            "class_weight": "balanced",
        })

        mlflow.log_params({
            "n_features_antes_onehot": len(features_modelo),
            "cv_folds": 5,
            "seed": SEED,
            **{
                f"modelo__{chave}": str(valor)
                for chave, valor in modelo.get_params().items()
            }
        })

        metricas_cv = {
            coluna: float(resultado[coluna])
            for coluna in resultado.index
            if coluna.endswith("_media") or coluna.endswith("_desvio")
        }

        mlflow.log_metrics(metricas_cv)

print("Baselines de cross-validation registrados no MLflow.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Avaliação do efeito de conjunto de features
# MAGIC
# MAGIC O objetivo é descobrir se determinadas features realmente ajudam cada modelo.
# MAGIC
# MAGIC Mantemos:
# MAGIC - Mesmo split treino/teste
# MAGIC - Mesmos 5 folds
# MAGIC - Mesmas métricas
# MAGIC - Mesmo class_weight="balanced"
# MAGIC

# COMMAND ----------


# ------------------------------------------------------------
# 1. Definir conjuntos de features
# ------------------------------------------------------------

features_base = features_modelo.copy()

conjuntos_features = {
    # "completo": features_base, ## esse é igaul ao baseline que já executou

    "sem_totalcharges": [
        c for c in features_base
        if c != "TotalCharges"
    ],

    "sem_flags_ausencia": [
        c for c in features_base
        if c not in ["tenure_missing", "tc_missing"]
    ],

    "pagamento_compacto": [
        c for c in features_base
        if c != "PaymentMethod"
    ] + ["usa_electronic_check", "pagamento_automatico"],
}

for nome, features in conjuntos_features.items():
    print(f"{nome}: {len(features)} features")

# COMMAND ----------

def montar_pipeline(tipo_modelo, X_referencia):
    numericas_local = X_referencia.select_dtypes(include=np.number).columns.tolist()
    categoricas_local = [
        c for c in X_referencia.columns
        if c not in numericas_local
    ]

    preprocessador_logistica_local = ColumnTransformer(
        transformers=[
            ("numericas", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numericas_local),

            ("categoricas", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categoricas_local),
        ],
        remainder="drop",
    )

    preprocessador_arvores_local = ColumnTransformer(
        transformers=[
            ("numericas",
             SimpleImputer(strategy="median"),
             numericas_local),

            ("categoricas", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categoricas_local),
        ],
        remainder="drop",
    )

    if tipo_modelo == "regressao_logistica":
        return Pipeline([
            ("preprocessamento", preprocessador_logistica_local),
            ("modelo", LogisticRegression(
                max_iter=2_000,
                class_weight="balanced",
                random_state=SEED,
            )),
        ])

    if tipo_modelo == "random_forest":
        return Pipeline([
            ("preprocessamento", preprocessador_arvores_local),
            ("modelo", RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=1,
            )),
        ])

    raise ValueError("Modelo não reconhecido.")

# COMMAND ----------

mlflow.set_experiment(EXPERIMENTO)

resultados_features = []

for nome_conjunto, features in conjuntos_features.items():

    # Reconstroi treino usando exatamente os mesmos clientes
    # do split original. X_test continua sem uso.
    X_train_variante = history.loc[X_train.index, features].copy()

    for nome_modelo in ["regressao_logistica", "random_forest"]:

        pipeline = montar_pipeline(nome_modelo, X_train_variante)

        resultado = cross_validate(
            pipeline,
            X_train_variante,
            y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
        )

        linha = {
            "modelo": nome_modelo,
            "conjunto_features": nome_conjunto,
            "n_features": len(features),
        }

        for metrica in scoring:
            linha[f"{metrica}_media"] = resultado[f"test_{metrica}"].mean()
            linha[f"{metrica}_desvio"] = resultado[f"test_{metrica}"].std()

        resultados_features.append(linha)

        # Registro de uma run por combinação modelo + features
        with mlflow.start_run(
            run_name=f"features_cv__{nome_modelo}__{nome_conjunto}"
        ):
            mlflow.set_tags({
                "etapa": "features_cv",
                "modelo": nome_modelo,
                "conjunto_features": nome_conjunto,
                "class_weight": "balanced",
                "pagamento": (
                    "compacto"
                    if nome_conjunto == "pagamento_compacto"
                    else "PaymentMethod_completo"
                ),
            })

            mlflow.log_params({
                "n_features_antes_onehot": len(features),
                "features": ", ".join(features),
                "cv_folds": 5,
                "seed": SEED,
                **{
                    f"modelo__{chave}": str(valor)
                    for chave, valor
                    in pipeline.named_steps["modelo"].get_params().items()
                },
            })

            mlflow.log_metrics({
                chave: float(valor)
                for chave, valor in linha.items()
                if chave.endswith("_media") or chave.endswith("_desvio")
            })


resultados_features = (
    pd.DataFrame(resultados_features)
    .sort_values(
        ["modelo", "pr_auc_media"],
        ascending=[True, False]
    )
    .reset_index(drop=True)
)

display(
    resultados_features.style.format({
        coluna: "{:.3f}"
        for coluna in resultados_features.columns
        if coluna not in ["modelo", "conjunto_features", "n_features"]
    })
)

# COMMAND ----------

# MAGIC %md
# MAGIC As variações de features mudaram muito pouco o desempenho. Isso aponta para modelos que parecem estáveis.
# MAGIC Os modelos com conjunto de dados que exclui as flags de ausência de tenure e total charge tiveram melhoras nas métricas de Random Forest, e melhor valor de PR_AUC em Regressão Logística.<br>
# MAGIC COmo sua exclusão não prejudica as métricas e simplifica o modelo, podemos partir dessa premissa para testar parâmetros dos modelos em etapas seguintes.

# COMMAND ----------

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# Conjunto escolhido para seguir ao tuning:
# mantém TotalCharges e PaymentMethod;
# remove as flags de ausência.
features_importancia = [
    c for c in features_modelo
    if c not in ["tenure_missing", "tc_missing"]
]

# Usa somente os clientes que já pertencem ao treino.
X_importancia = history.loc[X_train.index, features_importancia].copy()
y_importancia = y_train.copy()

# Cria uma validação interna.
# X_test continua totalmente intocado.
X_fit, X_validacao, y_fit, y_validacao = train_test_split(
    X_importancia,
    y_importancia,
    test_size=0.20,
    stratify=y_importancia,
    random_state=SEED,
)

resultados_importancia = []

for nome_modelo in ["regressao_logistica", "random_forest"]:

    # Reutiliza a função criada no Passo 1.
    pipeline = montar_pipeline(nome_modelo, X_fit)

    pipeline.fit(X_fit, y_fit)

    permutacao = permutation_importance(
        pipeline,
        X_validacao,
        y_validacao,
        scoring="average_precision",  # PR-AUC
        n_repeats=20,
        random_state=SEED,
        n_jobs=-1,
    )

    tabela = pd.DataFrame({
        "feature": features_importancia,
        "importancia_media": permutacao.importances_mean,
        "importancia_desvio": permutacao.importances_std,
        "modelo": nome_modelo,
    }).sort_values("importancia_media", ascending=False)

    resultados_importancia.append(tabela)

importancias = pd.concat(resultados_importancia, ignore_index=True)

display(
    importancias
    .sort_values(["modelo", "importancia_media"], ascending=[True, False])
    .style.format({
        "importancia_media": "{:.4f}",
        "importancia_desvio": "{:.4f}",
    })
)

# COMMAND ----------

# MAGIC %md
# MAGIC Olhando a importância das variáveis para os modelos através de Permutation Importance, vemos que o risco de churn é explicado principalmente pelo estágio do relacionamento (tenure), pelo compromisso contratual (Contract), pelo tipo de serviço de internet e pela estrutura de cobrança acumulada (TotalCharges).
# MAGIC
# MAGIC Observando essas estruturas, faremos mais dois testes de conjunto de feature:
# MAGIC - a exclusão de features derivadas
# MAGIC - e o modelo mais enxuto, conservando features que se mostraram relevantes na EDA e na feature importance.

# COMMAND ----------

# Base escolhida após o primeiro teste de features:
features_sem_flags = [
    c for c in features_modelo
    if c not in ["tenure_missing", "tc_missing"]
]

# 1. Remove apenas features derivadas redundantes.
features_sem_derivadas = [
    c for c in features_sem_flags
    if c not in [
        "tem_internet",
        "fatura_sobre_renda",
        "qtd_servicos_adicionais",
    ]
]

# 2. Versão mais enxuta:
# remove derivadas e atributos com sinal muito fraco nos dois modelos.
features_modelo_enxuto = [
    c for c in features_sem_derivadas
    if c not in [
        "Partner",
        "Dependents",
        "StreamingMovies",
        "StreamingTV",
    ]
]

conjuntos_adicionais = {
    "sem_features_derivadas": features_sem_derivadas,
    "modelo_enxuto": features_modelo_enxuto,
}

for nome, features in conjuntos_adicionais.items():
    print(f"\n{nome}: {len(features)} features")
    print(features)

# COMMAND ----------

mlflow.set_experiment(EXPERIMENTO)

resultados_adicionais = []

for nome_conjunto, features in conjuntos_adicionais.items():

    # Usa exatamente os mesmos clientes do treino original.
    X_train_variante = history.loc[X_train.index, features].copy()

    for nome_modelo in ["regressao_logistica", "random_forest"]:

        pipeline = montar_pipeline(nome_modelo, X_train_variante)

        resultado = cross_validate(
            pipeline,
            X_train_variante,
            y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
        )

        linha = {
            "modelo": nome_modelo,
            "conjunto_features": nome_conjunto,
            "n_features": len(features),
        }

        for metrica in scoring:
            linha[f"{metrica}_media"] = resultado[f"test_{metrica}"].mean()
            linha[f"{metrica}_desvio"] = resultado[f"test_{metrica}"].std()

        resultados_adicionais.append(linha)

        with mlflow.start_run(
            run_name=f"features_cv_round2__{nome_modelo}__{nome_conjunto}"
        ):
            mlflow.set_tags({
                "etapa": "features_cv_round2",
                "modelo": nome_modelo,
                "conjunto_features": nome_conjunto,
                "class_weight": "balanced",
                "pagamento": "PaymentMethod_completo",
            })

            mlflow.log_params({
                "n_features_antes_onehot": len(features),
                "features": ", ".join(features),
                "cv_folds": 5,
                "seed": SEED,
                **{
                    f"modelo__{chave}": str(valor)
                    for chave, valor
                    in pipeline.named_steps["modelo"].get_params().items()
                },
            })

            mlflow.log_metrics({
                chave: float(valor)
                for chave, valor in linha.items()
                if chave.endswith("_media") or chave.endswith("_desvio")
            })


resultados_adicionais = pd.DataFrame(resultados_adicionais)

display(
    resultados_adicionais
    .sort_values(["modelo", "pr_auc_media"], ascending=[True, False])
    .style.format({
        coluna: "{:.3f}"
        for coluna in resultados_adicionais.columns
        if coluna not in ["modelo", "conjunto_features", "n_features"]
    })
)

# COMMAND ----------

baseline_sem_flags = resultados_features[
    resultados_features["conjunto_features"] == "sem_flags_ausencia"
].copy()

comparacao_features = pd.concat(
    [baseline_sem_flags, resultados_adicionais],
    ignore_index=True,
)

display(
    comparacao_features
    .sort_values(["modelo", "pr_auc_media"], ascending=[True, False])
    .style.format({
        coluna: "{:.3f}"
        for coluna in comparacao_features.columns
        if coluna not in ["modelo", "conjunto_features", "n_features"]
    })
)

# COMMAND ----------

# MAGIC %md
# MAGIC Novamente, as métricas virtualmente empatam.<br>
# MAGIC Observa-se um ganho marginal de recall e F2 em regressão logística olhando para o conjunto de features mais enxuto.<br>
# MAGIC Pode ser um benefício remover essas 7 variáveis do modelo, conferindo mais estabilidade e interpretabilidade.<br>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Avaliação de ajustes de hiperparâmetros
# MAGIC
# MAGIC Como o espaço de busca é pequeno, Grid Search manual com MLflow é mais transparente e com custo computacional possível.

# COMMAND ----------

## Preparando a base de treino com as 14 features selecionadas na etapa anterior
features_tuning = features_modelo_enxuto

X_train_tuning = history.loc[X_train.index, features_tuning].copy()

print(f"{len(features_tuning)} features:")
print(features_tuning)

# COMMAND ----------

## Definição de hiperparâmetros a seren testados

from sklearn.model_selection import ParameterGrid

grid_logistica = list(ParameterGrid({
    "C": [0.05, 0.1, 0.3, 1.0, 3.0, 10.0]
}))

grid_random_forest = list(ParameterGrid({
    "max_depth": [6, 10, None],
    "min_samples_leaf": [5, 10, 20],
}))

print("Configurações da Logística:", len(grid_logistica))
print("Configurações da Random Forest:", len(grid_random_forest))

# COMMAND ----------

mlflow.set_experiment(EXPERIMENTO)

resultados_tuning = []


def executar_tuning(nome_modelo, configuracoes):
    resultados_modelo = []

    for config in configuracoes:

        pipeline = montar_pipeline(nome_modelo, X_train_tuning)

        # Aplica a configuração desta tentativa
        if nome_modelo == "regressao_logistica":
            pipeline.set_params(
                modelo__C=config["C"]
            )

        elif nome_modelo == "random_forest":
            pipeline.set_params(
                modelo__max_depth=config["max_depth"],
                modelo__min_samples_leaf=config["min_samples_leaf"],
            )

        resultado_cv = cross_validate(
            pipeline,
            X_train_tuning,
            y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
        )

        linha = {
            "modelo": nome_modelo,
            **config,
            "n_features": len(features_tuning),
        }

        for metrica in scoring:
            linha[f"{metrica}_media"] = resultado_cv[f"test_{metrica}"].mean()
            linha[f"{metrica}_desvio"] = resultado_cv[f"test_{metrica}"].std()

        resultados_modelo.append(linha)

        # Uma run do MLflow por combinação de hiperparâmetros
        config_texto = "__".join(
            f"{chave}={valor}"
            for chave, valor in config.items()
        )

        with mlflow.start_run(
            run_name=f"tuning_cv__{nome_modelo}__{config_texto}"
        ):
            mlflow.set_tags({
                "etapa": "tuning_cv",
                "modelo": nome_modelo,
                "conjunto_features": "modelo_enxuto",
                "class_weight": "balanced",
            })

            mlflow.log_params({
                "n_features_antes_onehot": len(features_tuning),
                "features": ", ".join(features_tuning),
                "cv_folds": 5,
                "seed": SEED,
                **{
                    f"tuning__{chave}": str(valor)
                    for chave, valor in config.items()
                },
            })

            mlflow.log_metrics({
                chave: float(valor)
                for chave, valor in linha.items()
                if chave.endswith("_media") or chave.endswith("_desvio")
            })

    return resultados_modelo


resultados_tuning += executar_tuning(
    "regressao_logistica",
    grid_logistica
)

resultados_tuning += executar_tuning(
    "random_forest",
    grid_random_forest
)

resultados_tuning = pd.DataFrame(resultados_tuning)

# COMMAND ----------

melhores_logistica = (
    resultados_tuning
    .query("modelo == 'regressao_logistica'")
    .sort_values(
        ["pr_auc_media", "f2_media"],
        ascending=False
    )
)

melhores_random_forest = (
    resultados_tuning
    .query("modelo == 'random_forest'")
    .sort_values(
        ["pr_auc_media", "f2_media"],
        ascending=False
    )
)

print("Melhores configurações — Regressão Logística")
display(
    melhores_logistica.style.format({
        coluna: "{:.3f}"
        for coluna in melhores_logistica.columns
        if coluna not in ["modelo", "C", "n_features"]
    })
)

print("Melhores configurações — Random Forest")
display(
    melhores_random_forest.style.format({
        coluna: "{:.3f}"
        for coluna in melhores_random_forest.columns
        if coluna not in [
            "modelo",
            "max_depth",
            "min_samples_leaf",
            "n_features",
        ]
    })
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Seleção dos finalistas antes do boosting
# MAGIC
# MAGIC A Regressão Logística e a Random Forest foram escolhidas como finalistas após comparação por cross-validation com cinco folds, usando PR-AUC como métrica principal e F2 como critério de desempate.
# MAGIC
# MAGIC A Regressão Logística com `C=0,3` foi selecionada por manter a melhor PR-AUC observada (`0,657`), alto recall (`0,796`) e F2 (`0,717`) com regularização mais forte e maior interpretabilidade. A configuração usa o conjunto enxuto de 14 features, sem perda material de desempenho frente às versões mais completas.
# MAGIC
# MAGIC A Random Forest com `max_depth=10` e `min_samples_leaf=20` foi selecionada como challenger de árvores. Sua PR-AUC (`0,654`) foi praticamente equivalente à melhor configuração da floresta, enquanto o maior tamanho mínimo de folha trouxe maior regularização, recall (`0,772`) e F2 (`0,707`) mais altos.
# MAGIC
# MAGIC Em seguida, XGBoost será testado como um último challenger. Ele pode capturar interações e relações não lineares de forma mais eficiente que a Random Forest. Será mantido apenas se apresentar ganho consistente de desempenho em cross-validation, sem perda relevante de estabilidade ou interpretabilidade prática.
# MAGIC

# COMMAND ----------

# MAGIC %pip install xgboost
# MAGIC from xgboost import XGBClassifier
# MAGIC
# MAGIC print("XGBoost importado com sucesso.")

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


# Usa exatamente o conjunto enxuto selecionado.
features_xgb = features_modelo_enxuto

# Mesmos clientes do treino original.
X_train_xgb = history.loc[X_train.index, features_xgb].copy()
y_train_xgb = y_train.copy()

numericas_xgb = X_train_xgb.select_dtypes(include=np.number).columns.tolist()
categoricas_xgb = [
    coluna for coluna in X_train_xgb.columns
    if coluna not in numericas_xgb
]

# Mesmo tratamento categórico usado nos demais modelos de árvore.
preprocessador_xgb = ColumnTransformer(
    transformers=[
        (
            "numericas",
            SimpleImputer(strategy="median"),
            numericas_xgb,
        ),
        (
            "categoricas",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categoricas_xgb,
        ),
    ],
    remainder="drop",
)

# Balanceamento equivalente à ideia de class_weight="balanced".
# Ex.: se há ~2,7 não-churners para cada churner, o erro em churn
# recebe aproximadamente 2,7 vezes mais peso no treinamento.
scale_pos_weight = (
    (y_train_xgb == 0).sum()
    / (y_train_xgb == 1).sum()
)

print(f"scale_pos_weight: {scale_pos_weight:.3f}")

xgb_challenger = Pipeline([
    ("preprocessamento", preprocessador_xgb),
    ("modelo", XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=10,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=1,
        verbosity=0,
    )),
])

resultado_xgb = cross_validate(
    xgb_challenger,
    X_train_xgb,
    y_train_xgb,
    scoring=scoring,
    cv=cv,
    n_jobs=-1,
    return_train_score=False,
)

linha_xgb = {
    "modelo": "xgboost_challenger",
    "conjunto_features": "modelo_enxuto",
    "n_features": len(features_xgb),
}

for metrica in scoring:
    linha_xgb[f"{metrica}_media"] = resultado_xgb[f"test_{metrica}"].mean()
    linha_xgb[f"{metrica}_desvio"] = resultado_xgb[f"test_{metrica}"].std()

resultado_xgb_df = pd.DataFrame([linha_xgb])

display(
    resultado_xgb_df.style.format({
        coluna: "{:.3f}"
        for coluna in resultado_xgb_df.columns
        if coluna not in ["modelo", "conjunto_features", "n_features"]
    })
)

# COMMAND ----------

mlflow.set_experiment(EXPERIMENTO)

with mlflow.start_run(run_name="challenger_cv__xgboost__modelo_enxuto"):

    mlflow.set_tags({
        "etapa": "challenger_cv",
        "modelo": "xgboost",
        "conjunto_features": "modelo_enxuto",
        "balanceamento": "scale_pos_weight",
    })

    mlflow.log_params({
        "n_features_antes_onehot": len(features_xgb),
        "features": ", ".join(features_xgb),
        "cv_folds": 5,
        "seed": SEED,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_weight": 10,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
    })

    mlflow.log_metrics({
        chave: float(valor)
        for chave, valor in linha_xgb.items()
        if chave.endswith("_media") or chave.endswith("_desvio")
    })

print("XGBoost registrado no MLflow.")

# COMMAND ----------

# MAGIC %md
# MAGIC Por último um teste com CatBoost

# COMMAND ----------

# MAGIC %pip install catboost

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


# ------------------------------------------------------------
# Configurações
# ------------------------------------------------------------

CATALOG = "workspace.default"
TABELA_HISTORY = f"{CATALOG}.history_modelagem"
TABELA_INFERENCE = f"{CATALOG}.inference_modelagem"

EXPERIMENTO = "/Shared/terra_signal_churn"
TARGET = "Churn"
SEED = 42


# ------------------------------------------------------------
# Dados
# ------------------------------------------------------------

history = spark.table(TABELA_HISTORY).toPandas()
inference = spark.table(TABELA_INFERENCE).toPandas()

ATRIBUTOS_SENSIVEIS = ["gender", "SeniorCitizen"]
FLAGS_PAGAMENTO = ["usa_electronic_check", "pagamento_automatico"]

# Cenário original de pagamento completo.
features_modelo = [
    coluna for coluna in inference.columns
    if coluna not in ATRIBUTOS_SENSIVEIS + FLAGS_PAGAMENTO
]

# Conjunto final enxuto, com 14 features.
features_modelo_enxuto = [
    coluna for coluna in features_modelo
    if coluna not in [
        "tenure_missing",
        "tc_missing",
        "tem_internet",
        "fatura_sobre_renda",
        "qtd_servicos_adicionais",
        "Partner",
        "Dependents",
        "StreamingMovies",
        "StreamingTV",
    ]
]

X = history[features_modelo].copy()
y = history[TARGET].astype(int).copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEED,
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

scoring = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "f2": make_scorer(fbeta_score, beta=2),
}

mlflow.set_experiment(EXPERIMENTO)

print("Recuperação concluída.")
print(f"history: {history.shape}")
print(f"treino: {X_train.shape}")
print(f"teste reservado: {X_test.shape}")
print(f"features enxutas: {len(features_modelo_enxuto)}")
print(features_modelo_enxuto)

# COMMAND ----------

from catboost import CatBoostClassifier
from sklearn.model_selection import cross_validate

# Mesmas 14 features do modelo enxuto.
features_catboost = features_modelo_enxuto

# Mesmos clientes do treino original.
X_train_catboost = history.loc[X_train.index, features_catboost].copy()
y_train_catboost = y_train.copy()

# CatBoost recebe categorias como texto.
categoricas_catboost = [
    coluna
    for coluna in X_train_catboost.columns
    if X_train_catboost[coluna].dtype == "object"
]

for coluna in categoricas_catboost:
    X_train_catboost[coluna] = (
        X_train_catboost[coluna]
        .fillna("__MISSING__")
        .astype(str)
    )

print("Features categóricas:")
print(categoricas_catboost)

catboost_challenger = CatBoostClassifier(
    iterations=400,
    learning_rate=0.05,
    depth=5,
    l2_leaf_reg=5.0,
    auto_class_weights="Balanced",
    loss_function="Logloss",
    random_seed=SEED,
    thread_count=1,
    verbose=False,
    allow_writing_files=False,
)

resultado_catboost = cross_validate(
    catboost_challenger,
    X_train_catboost,
    y_train_catboost,
    scoring=scoring,
    cv=cv,
    n_jobs=-1,
    return_train_score=False,
    params={
        "cat_features": categoricas_catboost
    },
)

linha_catboost = {
    "modelo": "catboost_challenger",
    "conjunto_features": "modelo_enxuto",
    "n_features": len(features_catboost),
}

for metrica in scoring:
    linha_catboost[f"{metrica}_media"] = (
        resultado_catboost[f"test_{metrica}"].mean()
    )
    linha_catboost[f"{metrica}_desvio"] = (
        resultado_catboost[f"test_{metrica}"].std()
    )

resultado_catboost_df = pd.DataFrame([linha_catboost])

display(
    resultado_catboost_df.style.format({
        coluna: "{:.3f}"
        for coluna in resultado_catboost_df.columns
        if coluna not in ["modelo", "conjunto_features", "n_features"]
    })
)

# COMMAND ----------

from sklearn.base import clone

resultados_folds = {
    metrica: []
    for metrica in scoring
}

for indice_treino, indice_validacao in cv.split(
    X_train_catboost,
    y_train_catboost
):
    X_fold_train = X_train_catboost.iloc[indice_treino]
    X_fold_validacao = X_train_catboost.iloc[indice_validacao]

    y_fold_train = y_train_catboost.iloc[indice_treino]
    y_fold_validacao = y_train_catboost.iloc[indice_validacao]

    modelo_fold = clone(catboost_challenger)

    modelo_fold.fit(
        X_fold_train,
        y_fold_train,
        cat_features=categoricas_catboost,
    )

    probabilidades = modelo_fold.predict_proba(X_fold_validacao)[:, 1]
    previsoes = (probabilidades >= 0.50).astype(int)

    resultados_folds["roc_auc"].append(
        roc_auc_score(y_fold_validacao, probabilidades)
    )
    resultados_folds["pr_auc"].append(
        average_precision_score(y_fold_validacao, probabilidades)
    )
    resultados_folds["precision"].append(
        precision_score(y_fold_validacao, previsoes, zero_division=0)
    )
    resultados_folds["recall"].append(
        recall_score(y_fold_validacao, previsoes, zero_division=0)
    )
    resultados_folds["f1"].append(
        f1_score(y_fold_validacao, previsoes, zero_division=0)
    )
    resultados_folds["f2"].append(
        fbeta_score(y_fold_validacao, previsoes, beta=2, zero_division=0)
    )

linha_catboost = {
    "modelo": "catboost_challenger",
    "conjunto_features": "modelo_enxuto",
    "n_features": len(features_catboost),
}

for metrica, valores in resultados_folds.items():
    linha_catboost[f"{metrica}_media"] = np.mean(valores)
    linha_catboost[f"{metrica}_desvio"] = np.std(valores)

resultado_catboost_df = pd.DataFrame([linha_catboost])

display(resultado_catboost_df)

# COMMAND ----------

mlflow.set_experiment(EXPERIMENTO)

with mlflow.start_run(run_name="challenger_cv__catboost__modelo_enxuto"):

    mlflow.set_tags({
        "etapa": "challenger_cv",
        "modelo": "catboost",
        "conjunto_features": "modelo_enxuto",
        "balanceamento": "auto_class_weights_balanced",
        "categoricas": "nativas_sem_onehot",
    })

    mlflow.log_params({
        "n_features": len(features_catboost),
        "features": ", ".join(features_catboost),
        "cv_folds": 5,
        "seed": SEED,
        "iterations": 400,
        "learning_rate": 0.05,
        "depth": 5,
        "l2_leaf_reg": 5.0,
        "auto_class_weights": "Balanced",
    })

    mlflow.log_metrics({
        chave: float(valor)
        for chave, valor in linha_catboost.items()
        if chave.endswith("_media") or chave.endswith("_desvio")
    })

# COMMAND ----------

# MAGIC %md
# MAGIC Testando também o threshold da regressâo logística - considerando a amostra desbalanceada, o threshold 0.5 automático não necessariamente é a melhor escolha.<br>
# MAGIC Vamos varrer thresholds de 0.2 a 0.8 e avaliar as métricas resultantes.

# COMMAND ----------

import numpy as np
import pandas as pd

from sklearn.metrics import make_scorer, fbeta_score
from sklearn.model_selection import StratifiedKFold, train_test_split


# Configuração
CATALOG = "workspace.default"
TABELA_HISTORY = f"{CATALOG}.history_modelagem"

TARGET = "Churn"
SEED = 42


# Carregar apenas a tabela necessária
history = spark.table(TABELA_HISTORY).toPandas()

ATRIBUTOS_SENSIVEIS = ["gender", "SeniorCitizen"]
FLAGS_PAGAMENTO = ["usa_electronic_check", "pagamento_automatico"]

# Cenário de pagamento completo:
# mantém PaymentMethod e exclui as flags derivadas.
features_modelo = [
    coluna
    for coluna in history.columns
    if coluna not in [TARGET] + ATRIBUTOS_SENSIVEIS + FLAGS_PAGAMENTO
]

# Sem flags de ausência
features_sem_flags = [
    coluna
    for coluna in features_modelo
    if coluna not in ["tenure_missing", "tc_missing"]
]

# Sem derivadas redundantes
features_sem_derivadas = [
    coluna
    for coluna in features_sem_flags
    if coluna not in [
        "tem_internet",
        "fatura_sobre_renda",
        "qtd_servicos_adicionais",
    ]
]

# Modelo enxuto validado: 14 features
features_modelo_enxuto = [
    coluna
    for coluna in features_sem_derivadas
    if coluna not in [
        "Partner",
        "Dependents",
        "StreamingMovies",
        "StreamingTV",
    ]
]

assert len(features_modelo_enxuto) == 14, (
    f"Esperadas 14 features; encontrei {len(features_modelo_enxuto)}."
)

X = history[features_modelo].copy()
y = history[TARGET].astype(int).copy()

# Mesmo split usado nos testes anteriores
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEED,
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

scoring = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "f2": make_scorer(fbeta_score, beta=2),
}

print("Recuperação concluída.")
print(f"history: {history.shape}")
print(f"Treino: {X_train.shape}")
print(f"Teste reservado: {X_test.shape}")
print(f"Features enxutas: {len(features_modelo_enxuto)}")
print(features_modelo_enxuto)

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ------------------------------------------------------------
# 1. Dados e pipeline da melhor Regressão Logística
# ------------------------------------------------------------

assert len(features_modelo_enxuto) == 14, (
    f"Esperadas 14 features; encontradas {len(features_modelo_enxuto)}."
)

X_train_threshold = history.loc[
    X_train.index,
    features_modelo_enxuto
].copy()

y_train_threshold = y_train.copy()

numericas_threshold = X_train_threshold.select_dtypes(
    include=np.number
).columns.tolist()

categoricas_threshold = [
    coluna for coluna in X_train_threshold.columns
    if coluna not in numericas_threshold
]

preprocessador_threshold = ColumnTransformer(
    transformers=[
        (
            "numericas",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numericas_threshold,
        ),
        (
            "categoricas",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categoricas_threshold,
        ),
    ],
    remainder="drop",
)

logistica_final_cv = Pipeline([
    ("preprocessamento", preprocessador_threshold),
    ("modelo", LogisticRegression(
        C=0.3,
        max_iter=2_000,
        class_weight="balanced",
        random_state=SEED,
    )),
])


# ------------------------------------------------------------
# 2. Probabilidades out-of-fold no treino
# ------------------------------------------------------------
# Cada cliente recebe um score de um modelo que não foi treinado nele.

scores_oof = cross_val_predict(
    logistica_final_cv,
    X_train_threshold,
    y_train_threshold,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]

print("Probabilidades out-of-fold criadas.")
print(pd.Series(scores_oof).describe())


# ------------------------------------------------------------
# 3. Avaliar thresholds fixos
# ------------------------------------------------------------

def avaliar_threshold(y_real, scores, threshold):
    previsoes = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_real, previsoes).ravel()

    return {
        "threshold": threshold,
        "clientes_marcados": int(previsoes.sum()),
        "pct_clientes_marcados": previsoes.mean(),
        "precision": precision_score(y_real, previsoes, zero_division=0),
        "recall": recall_score(y_real, previsoes, zero_division=0),
        "f1": f1_score(y_real, previsoes, zero_division=0),
        "f2": fbeta_score(y_real, previsoes, beta=2, zero_division=0),
        "verdadeiros_positivos": int(tp),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdadeiros_negativos": int(tn),
    }


thresholds = np.round(np.arange(0.20, 0.81, 0.05), 2)

resultado_thresholds = pd.DataFrame([
    avaliar_threshold(y_train_threshold, scores_oof, threshold)
    for threshold in thresholds
])

display(resultado_thresholds)


# ------------------------------------------------------------
# 4. Avaliar top 10%, 20% e 30% por score
# ------------------------------------------------------------

resultados_top_n = []

for percentual in [0.10, 0.20, 0.30]:
    threshold_top_n = np.quantile(scores_oof, 1 - percentual)

    resultado = avaliar_threshold(
        y_train_threshold,
        scores_oof,
        threshold_top_n,
    )

    resultado["top_percentual"] = percentual
    resultado["threshold_equivalente"] = threshold_top_n

    resultados_top_n.append(resultado)

resultado_top_n = pd.DataFrame(resultados_top_n)

display(resultado_top_n)


# ------------------------------------------------------------
# 5. Gráfico de trade-off
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    resultado_thresholds["threshold"],
    resultado_thresholds["precision"],
    marker="o",
    label="Precision",
)

ax.plot(
    resultado_thresholds["threshold"],
    resultado_thresholds["recall"],
    marker="o",
    label="Recall",
)

ax.plot(
    resultado_thresholds["threshold"],
    resultado_thresholds["f2"],
    marker="o",
    label="F2",
)

ax.set_xlabel("Threshold")
ax.set_ylabel("Métrica")
ax.set_title("Trade-off de threshold — Logística C=0,3")
ax.grid(alpha=0.3)
ax.legend()

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Teste SVM

# COMMAND ----------



# COMMAND ----------

# SVM RBF como challenger
# Usa as mesmas 14 features, treino estratificado e CV de 5 folds.
# O conjunto de teste reservado NÃO é usado neste bloco.

import mlflow
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# COMMAND ----------

CATALOG = "workspace.default"
TABELA_HISTORY = f"{CATALOG}.history_modelagem"
TARGET = "Churn"
SEED = 42
EXPERIMENTO = "/Shared/terra_signal_churn"

# Mesmo conjunto enxuto usado nos melhores testes anteriores.
features_modelo_enxuto = [
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "MonthlyIncome",
]

NUMERICAS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "MonthlyIncome",
]

CATEGORICAS = [
    coluna for coluna in features_modelo_enxuto
    if coluna not in NUMERICAS
]

history_svm = spark.table(TABELA_HISTORY).toPandas()

X = history_svm[features_modelo_enxuto].copy()
y = history_svm[TARGET].astype(int).copy()

# Recria exatamente a lógica de divisão estratificada.
# X_test fica reservado e não será usado no experimento abaixo.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEED,
)

assert len(features_modelo_enxuto) == 14
assert X_train.shape[1] == 14

print(f"Treino: {X_train.shape}")
print(f"Teste reservado, não utilizado: {X_test.shape}")
print(f"Taxa de churn no treino: {y_train.mean():.3%}")

# COMMAND ----------

# Pré-processamento adequado a SVM:
# - imputação de ausentes;
# - escala das numéricas;
# - one-hot encoding das categóricas.

preprocessador_svm = ColumnTransformer(
    transformers=[
        (
            "numericas",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                # with_mean=False mantém compatibilidade com a matriz esparsa
                # produzida pelo one-hot encoding.
                ("scaler", StandardScaler(with_mean=False)),
            ]),
            NUMERICAS,
        ),
        (
            "categoricas",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            CATEGORICAS,
        ),
    ],
    remainder="drop",
)

# RBF consegue capturar relações não lineares.
# probability=True permite usar predict_proba depois, caso o SVM seja competitivo.
svm_rbf = Pipeline([
    ("preprocessamento", preprocessador_svm),
    ("modelo", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=SEED,
    )),
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

scoring = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "f2": make_scorer(fbeta_score, beta=2, zero_division=0),
}

# COMMAND ----------

# Avaliação por CV e registro no MLflow.
# Este bloco cria uma única run nova.

resultado_svm = cross_validate(
    estimator=svm_rbf,
    X=X_train,
    y=y_train,
    cv=cv,
    scoring=scoring,
    return_train_score=False,
    n_jobs=-1,
)

metricas_svm = {
    f"{metrica}_media": float(np.mean(resultado_svm[f"test_{metrica}"]))
    for metrica in scoring
}

metricas_svm.update({
    f"{metrica}_desvio": float(np.std(resultado_svm[f"test_{metrica}"]))
    for metrica in scoring
})

mlflow.set_experiment(EXPERIMENTO)

with mlflow.start_run(
    run_name="challenger_cv__svm_rbf__modelo_enxuto"
) as run:
    mlflow.set_tags({
        "projeto": "terra_signal_churn",
        "etapa": "challenger_cv",
        "conjunto_features": "modelo_enxuto",
        "target": TARGET,
        "features_sensiveis_incluidas": "nao",
    })

    mlflow.log_params({
        "modelo": "svm_rbf",
        "features": ", ".join(features_modelo_enxuto),
        "n_features": len(features_modelo_enxuto),
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "class_weight": "balanced",
        "probability": True,
        "cv_folds": 5,
        "seed": SEED,
    })

    mlflow.log_metrics(metricas_svm)

print("Run registrada:", run.info.run_id)

resultado_svm_tabela = pd.DataFrame([{
    "modelo": "svm_rbf",
    "conjunto_features": "modelo_enxuto",
    "n_features": len(features_modelo_enxuto),
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    **metricas_svm,
}])

display(resultado_svm_tabela.round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Teste de Sendibilidade com variáveis de gênero e seniorCitizen

# COMMAND ----------

# Análise de sensibilidade:
# incluir gender e SeniorCitizen APENAS para comparar desempenho.

import mlflow
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATALOG = "workspace.default"
TABELA_HISTORY = f"{CATALOG}.history_modelagem"
TARGET = "Churn"
SEED = 42
EXPERIMENTO = "/Shared/terra_signal_churn"

features_modelo_enxuto = [
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "MonthlyIncome",
]

ATRIBUTOS_SENSIVEIS = ["gender", "SeniorCitizen"]

features_com_sensiveis = [
    *features_modelo_enxuto,
    *ATRIBUTOS_SENSIVEIS,
]

NUMERICAS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "MonthlyIncome",
]

# SeniorCitizen será tratado como categoria 0/1.
CATEGORICAS = [
    coluna for coluna in features_com_sensiveis
    if coluna not in NUMERICAS
]

history_auditoria = spark.table(TABELA_HISTORY).toPandas()

X = history_auditoria[features_com_sensiveis].copy()
y = history_auditoria[TARGET].astype(int).copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEED,
)

assert len(features_com_sensiveis) == 16
assert not X_train.isna().all().any()

print(f"Features sem atributos sensíveis: {len(features_modelo_enxuto)}")
print(f"Features com atributos sensíveis: {len(features_com_sensiveis)}")
print("Atributos incluídos apenas neste teste:", ATRIBUTOS_SENSIVEIS)
print(f"Treino: {X_train.shape}")
print(f"Teste reservado, não utilizado: {X_test.shape}")

# COMMAND ----------

# Pré-processadores

preprocessador_logistica_sensivel = ColumnTransformer(
    transformers=[
        (
            "numericas",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            NUMERICAS,
        ),
        (
            "categoricas",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            CATEGORICAS,
        ),
    ],
    remainder="drop",
)

preprocessador_arvores_sensivel = ColumnTransformer(
    transformers=[
        (
            "numericas",
            SimpleImputer(strategy="median"),
            NUMERICAS,
        ),
        (
            "categoricas",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            CATEGORICAS,
        ),
    ],
    remainder="drop",
)

modelos_sensibilidade = {
    "regressao_logistica_com_sensiveis": Pipeline([
        ("preprocessamento", preprocessador_logistica_sensivel),
        ("modelo", LogisticRegression(
            C=0.3,
            max_iter=2_000,
            class_weight="balanced",
            random_state=SEED,
        )),
    ]),
    "random_forest_com_sensiveis": Pipeline([
        ("preprocessamento", preprocessador_arvores_sensivel),
        ("modelo", RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=1,
        )),
    ]),
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

scoring = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "f2": make_scorer(fbeta_score, beta=2, zero_division=0),
}

# COMMAND ----------

# Cross-validation e registro no MLflow.
# Cria duas runs novas, identificadas como auditoria sensível.

mlflow.set_experiment(EXPERIMENTO)

resultados_sensiveis = []

for nome, pipeline in modelos_sensibilidade.items():

    resultado_cv = cross_validate(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    metricas = {
        f"{metrica}_media": float(
            np.mean(resultado_cv[f"test_{metrica}"])
        )
        for metrica in scoring
    }

    metricas.update({
        f"{metrica}_desvio": float(
            np.std(resultado_cv[f"test_{metrica}"])
        )
        for metrica in scoring
    })

    with mlflow.start_run(
        run_name=f"auditoria_sensivel_cv__{nome}"
    ) as run:
        mlflow.set_tags({
            "projeto": "terra_signal_churn",
            "etapa": "auditoria_sensivel_cv",
            "finalidade": "analise_de_sensibilidade_nao_producao",
            "atributos_sensiveis_incluidos": "gender,SeniorCitizen",
            "target": TARGET,
        })

        mlflow.log_params({
            "modelo": nome,
            "conjunto_features": "modelo_enxuto_com_sensiveis",
            "features": ", ".join(features_com_sensiveis),
            "n_features": len(features_com_sensiveis),
            "atributos_sensiveis": ", ".join(ATRIBUTOS_SENSIVEIS),
            "class_weight": "balanced",
            "cv_folds": 5,
            "seed": SEED,
        })

        # Parâmetros próprios de cada modelo
        if "logistica" in nome:
            mlflow.log_params({
                "C": 0.3,
                "max_iter": 2_000,
            })
        else:
            mlflow.log_params({
                "n_estimators": 400,
                "max_depth": 10,
                "min_samples_leaf": 20,
            })

        mlflow.log_metrics(metricas)

    resultados_sensiveis.append({
        "modelo": nome,
        "conjunto_features": "modelo_enxuto_com_sensiveis",
        "n_features": len(features_com_sensiveis),
        **metricas,
        "run_id": run.info.run_id,
    })

comparacao_sensiveis = pd.DataFrame(resultados_sensiveis)

display(
    comparacao_sensiveis.sort_values(
        ["pr_auc_media", "f2_media"],
        ascending=False,
    ).round(3)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Avaliação dos modelos testados

# COMMAND ----------

import mlflow
import pandas as pd
import numpy as np

EXPERIMENTO = "/Shared/terra_signal_churn"

experimento = mlflow.get_experiment_by_name(EXPERIMENTO)

if experimento is None:
    raise ValueError(f"Experimento não encontrado: {EXPERIMENTO}")

runs_mlflow = mlflow.search_runs(
    experiment_ids=[experimento.experiment_id],
    max_results=10_000,
    order_by=["attributes.start_time DESC"],
)

print(f"Runs encontradas: {len(runs_mlflow)}")
display(runs_mlflow.head())

# COMMAND ----------



# COMMAND ----------

import mlflow
import pandas as pd
import numpy as np

EXPERIMENTO = "/Shared/terra_signal_churn"

experimento = mlflow.get_experiment_by_name(EXPERIMENTO)

if experimento is None:
    raise ValueError(f"Experimento não encontrado: {EXPERIMENTO}")

runs_mlflow = mlflow.search_runs(
    experiment_ids=[experimento.experiment_id],
    max_results=10_000,
    order_by=["attributes.start_time DESC"],
)

print(f"Experimento: {experimento.name}")
print(f"Runs encontradas: {len(runs_mlflow)}")

# COMMAND ----------

colunas_diretas = [
    "run_id",
    "start_time",
    "tags.mlflow.runName",
    *sorted(c for c in runs_mlflow.columns if c.startswith("params.")),
    *sorted(c for c in runs_mlflow.columns if c.startswith("metrics.")),
]

tabela_direta_mlflow = (
    runs_mlflow[colunas_diretas]
    .sort_values("start_time", ascending=False)
    .reset_index(drop=True)
)

display(tabela_direta_mlflow)

# COMMAND ----------

colunas_parametros = sorted(
    c for c in runs_mlflow.columns
    if c.startswith("params.")
)

def resumir_parametros(linha):
    parametros = []

    for coluna in colunas_parametros:
        valor = linha[coluna]

        if pd.notna(valor):
            nome = coluna.replace("params.", "")
            parametros.append(f"{nome}={valor}")

    return " | ".join(parametros)

tabela_legivel = tabela_direta_mlflow.copy()

tabela_legivel["configuracao_registrada"] = (
    runs_mlflow.apply(resumir_parametros, axis=1)
)

colunas_metricas_principais = [
    "metrics.roc_auc_media",
    "metrics.roc_auc_desvio",
    "metrics.pr_auc_media",
    "metrics.pr_auc_desvio",
    "metrics.precision_media",
    "metrics.recall_media",
    "metrics.f1_media",
    "metrics.f2_media",
]

colunas_exibicao = [
    "tags.mlflow.runName",
    *[
        coluna
        for coluna in colunas_metricas_principais
        if coluna in tabela_legivel.columns
    ],
    "configuracao_registrada",
    "run_id",
]

display(
    tabela_legivel[colunas_exibicao]
    .sort_values("metrics.pr_auc_media", ascending=False)
)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Avaliação final no conjunto de teste
# MAGIC
# MAGIC O teste é usado uma única vez, depois da comparação via CV. As métricas abaixo
# MAGIC usam threshold padrão de 0,50 apenas como referência; o threshold de campanha
# MAGIC será escolhido no próximo bloco.

# COMMAND ----------

# ============================================================
# TESTE FINAL — executar uma única vez após congelar decisões
# ============================================================

import numpy as np
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
)

# ------------------------------------------------------------
# 1. Recuperar a mesma divisão treino/teste reservada
# ------------------------------------------------------------

CATALOG = "workspace.default"
TARGET = "Churn"
SEED = 42
EXPERIMENTO = "/Shared/terra_signal_churn"

history = spark.table(f"{CATALOG}.history_modelagem").toPandas()

features_modelo_enxuto = [
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "MonthlyIncome",
]

X = history[features_modelo_enxuto].copy()
y = history[TARGET].astype(int).copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEED,
)

print(f"Treino: {X_train.shape[0]:,} clientes")
print(f"Teste reservado: {X_test.shape[0]:,} clientes")
print(f"Taxa de churn no teste: {y_test.mean():.1%}")

# COMMAND ----------

# ------------------------------------------------------------
# 2. Definir os preprocessadores e os dois candidatos finais
# ------------------------------------------------------------

numericas = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "MonthlyIncome",
]

categoricas = [
    coluna for coluna in features_modelo_enxuto
    if coluna not in numericas
]

preprocessador_logistica = ColumnTransformer(
    transformers=[
        (
            "numericas",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                # with_mean=False preserva compatibilidade com matriz esparsa
                # produzida pelo One Hot Encoder.
                ("scaler", StandardScaler(with_mean=False)),
            ]),
            numericas,
        ),
        (
            "categoricas",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categoricas,
        ),
    ],
    remainder="drop",
)

preprocessador_arvores = ColumnTransformer(
    transformers=[
        (
            "numericas",
            SimpleImputer(strategy="median"),
            numericas,
        ),
        (
            "categoricas",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categoricas,
        ),
    ],
    remainder="drop",
)

modelos_finais = {
    "regressao_logistica_C_0_3": Pipeline([
        ("preprocessamento", preprocessador_logistica),
        ("modelo", LogisticRegression(
            C=0.3,
            max_iter=2_000,
            class_weight="balanced",
            random_state=SEED,
        )),
    ]),
    "random_forest_depth_10_leaf_20": Pipeline([
        ("preprocessamento", preprocessador_arvores),
        ("modelo", RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )),
    ]),
}

# COMMAND ----------

# ------------------------------------------------------------
# 3. Funções de avaliação
# ------------------------------------------------------------

def metricas_threshold(y_verdadeiro, probabilidades, threshold=0.5):
    previsoes = (probabilidades >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_verdadeiro, previsoes).ravel()

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_verdadeiro, probabilidades)),
        "pr_auc": float(average_precision_score(y_verdadeiro, probabilidades)),
        "precision": float(precision_score(y_verdadeiro, previsoes, zero_division=0)),
        "recall": float(recall_score(y_verdadeiro, previsoes, zero_division=0)),
        "f1": float(f1_score(y_verdadeiro, previsoes, zero_division=0)),
        "f2": float(fbeta_score(y_verdadeiro, previsoes, beta=2, zero_division=0)),
        "verdadeiros_positivos": int(tp),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdadeiros_negativos": int(tn),
        "clientes_marcados": int(previsoes.sum()),
        "pct_clientes_marcados": float(previsoes.mean()),
    }


def metricas_top_percentual(y_verdadeiro, probabilidades, top_percentual):
    """
    Avalia os clientes com os maiores scores.
    Ex.: Top 10% = 10% dos clientes do conjunto de teste com maior score.
    """
    avaliacao = pd.DataFrame({
        "churn_real": np.asarray(y_verdadeiro),
        "score_churn": probabilidades,
    }).sort_values("score_churn", ascending=False).reset_index(drop=True)

    n_top = max(1, int(np.ceil(len(avaliacao) * top_percentual)))
    top = avaliacao.head(n_top).copy()

    tp = int(top["churn_real"].sum())
    fp = int(len(top) - tp)
    total_churners = int(avaliacao["churn_real"].sum())
    fn = int(total_churners - tp)

    return {
        "top_percentual": float(top_percentual),
        "clientes_marcados": int(n_top),
        "pct_clientes_marcados": float(n_top / len(avaliacao)),
        "score_minimo_entrada": float(top["score_churn"].iloc[-1]),
        "precision": float(tp / n_top),
        "recall": float(tp / total_churners),
        "f1": float(f1_score(
            avaliacao["churn_real"],
            np.where(avaliacao.index < n_top, 1, 0),
            zero_division=0,
        )),
        "f2": float(fbeta_score(
            avaliacao["churn_real"],
            np.where(avaliacao.index < n_top, 1, 0),
            beta=2,
            zero_division=0,
        )),
        "verdadeiros_positivos": tp,
        "falsos_positivos": fp,
        "falsos_negativos": fn,
    }

# COMMAND ----------

# ------------------------------------------------------------
# 4. Executar o teste final e registrar no MLflow
# ------------------------------------------------------------

mlflow.set_experiment(EXPERIMENTO)

resultados_finais = []
resultados_top = []
pipelines_finais_ajustados = {}
scores_teste = {}

for nome_modelo, pipeline in modelos_finais.items():

    with mlflow.start_run(run_name=f"final_test__{nome_modelo}"):

        # Treina exclusivamente no conjunto de treino.
        pipeline.fit(X_train, y_train)

        # Avalia uma única vez no conjunto de teste reservado.
        probabilidades = pipeline.predict_proba(X_test)[:, 1]

        resultado_05 = metricas_threshold(
            y_test,
            probabilidades,
            threshold=0.50,
        )

        resultados_top_modelo = []
        for top_percentual in [0.10, 0.20, 0.30]:
            resultado_top = metricas_top_percentual(
                y_test,
                probabilidades,
                top_percentual,
            )
            resultado_top["modelo"] = nome_modelo
            resultados_top_modelo.append(resultado_top)
            resultados_top.append(resultado_top)

        # Salva resultados em memória para comparação e eventual escolha.
        pipelines_finais_ajustados[nome_modelo] = pipeline
        scores_teste[nome_modelo] = probabilidades

        linha_resultado = {
            "modelo": nome_modelo,
            "n_features": len(features_modelo_enxuto),
            **resultado_05,
        }
        resultados_finais.append(linha_resultado)

        # Metadados padronizados da run.
        mlflow.set_tags({
            "etapa": "final_test",
            "conjunto_features": "modelo_enxuto",
            "tipo_avaliacao": "teste_reservado",
            "observacao": (
                "Teste final dos dois candidatos congelados. "
                "Nao utilizar para novos ajustes."
            ),
        })

        mlflow.log_params({
            "modelo": nome_modelo,
            "n_features": len(features_modelo_enxuto),
            "features": ", ".join(features_modelo_enxuto),
            "seed": SEED,
            "test_size": 0.20,
            "threshold_referencia": 0.50,
        })

        # Parâmetros específicos de cada candidato.
        if nome_modelo == "regressao_logistica_C_0_3":
            mlflow.log_params({
                "C": 0.3,
                "class_weight": "balanced",
                "max_iter": 2000,
            })
        else:
            mlflow.log_params({
                "n_estimators": 400,
                "max_depth": 10,
                "min_samples_leaf": 20,
                "class_weight": "balanced",
            })

        mlflow.log_metrics({
            f"teste_{chave}": valor
            for chave, valor in resultado_05.items()
            if isinstance(valor, (int, float, np.integer, np.floating))
        })

        for resultado_top in resultados_top_modelo:
            faixa = f"top_{int(resultado_top['top_percentual'] * 100)}"
            mlflow.log_metrics({
                f"{faixa}_{chave}": valor
                for chave, valor in resultado_top.items()
                if chave != "top_percentual"
                and isinstance(valor, (int, float, np.integer, np.floating))
            })

        mlflow.log_dict(
            {
                "modelo": nome_modelo,
                "metricas_threshold_0_5": resultado_05,
                "metricas_top_percentual": resultados_top_modelo,
            },
            "resultados_teste_final.json",
        )

        print(f"Run final registrada: {nome_modelo}")

# COMMAND ----------

# ------------------------------------------------------------
# 5. Exibir a comparação para decisão
# ------------------------------------------------------------

comparacao_final = (
    pd.DataFrame(resultados_finais)
    [[
        "modelo",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "f2",
        "clientes_marcados",
        "pct_clientes_marcados",
        "verdadeiros_positivos",
        "falsos_positivos",
        "falsos_negativos",
    ]]
    .sort_values(
        ["pr_auc", "f2", "precision"],
        ascending=False,
    )
    .reset_index(drop=True)
)

comparacao_top_final = (
    pd.DataFrame(resultados_top)
    [[
        "modelo",
        "top_percentual",
        "clientes_marcados",
        "pct_clientes_marcados",
        "score_minimo_entrada",
        "precision",
        "recall",
        "f1",
        "f2",
        "verdadeiros_positivos",
        "falsos_positivos",
        "falsos_negativos",
    ]]
    .sort_values(["top_percentual", "precision"], ascending=[True, False])
    .reset_index(drop=True)
)

print("Comparação geral no teste reservado")
display(comparacao_final.round(3))

print("Comparação operacional por Top X%")
display(comparacao_top_final.round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC O Random Forest foi selecionado como modelo final por apresentar melhor desempenho global no conjunto de teste reservado, especialmente em PR-AUC, precision e F1. Também demonstrou maior concentração de churners no Top 10% da carteira, público prioritário para ações de retenção de maior custo. A Regressão Logística apresentou recall e F2 marginalmente superiores, mas a diferença foi pequena e não superou o ganho de precisão e ranking do Random Forest.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Treino do modelo em toda base histórica para inferência

# COMMAND ----------

# ============================================================
# MODELO FINAL — treino em toda a base histórica e scoring
# ============================================================

from sklearn.base import clone
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

CATALOG = "workspace.default"
TABELA_HISTORY = f"{CATALOG}.history_modelagem"
TABELA_INFERENCE = f"{CATALOG}.inference_tratado"
TABELA_SAIDA = f"{CATALOG}.clientes_risco_churn"

TARGET = "Churn"
SEED = 42
EXPERIMENTO = "/Shared/terra_signal_churn"

MODELO_ESCOLHIDO = "random_forest_depth_10_leaf_20"

# Base histórica completa: agora o modelo aprende também com o que
# antes estava reservado como teste
history_final = spark.table(TABELA_HISTORY).toPandas()

# Usamos inference_tratado porque ela preserva customerID para a operação.
inference_final = spark.table(TABELA_INFERENCE).toPandas()

X_history_final = history_final[features_modelo_enxuto].copy()
y_history_final = history_final[TARGET].astype(int).copy()
X_inference_final = inference_final[features_modelo_enxuto].copy()

assert "customerID" in inference_final.columns
assert inference_final["customerID"].notna().all()
assert inference_final["customerID"].is_unique
assert list(X_history_final.columns) == list(X_inference_final.columns)

print(f"Histórico para treino final: {X_history_final.shape}")
print(f"Carteira para score: {X_inference_final.shape}")
print(f"Modelo escolhido: {MODELO_ESCOLHIDO}")

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

numericas = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "MonthlyIncome",
]

categoricas = [
    coluna for coluna in features_modelo_enxuto
    if coluna not in numericas
]

preprocessador_arvores_final = ColumnTransformer(
    transformers=[
        (
            "numericas",
            SimpleImputer(strategy="median"),
            numericas,
        ),
        (
            "categoricas",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categoricas,
        ),
    ],
    remainder="drop",
)

modelo_final = Pipeline([
    ("preprocessamento", preprocessador_arvores_final),
    ("modelo", RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )),
])

print("Pipeline final do Random Forest recriado.")

# COMMAND ----------

# Treina o Random Forest final em toda a base histórica
# e calcula um score para cada cliente da carteira.

mlflow.set_experiment(EXPERIMENTO)

with mlflow.start_run(run_name="modelo_final__random_forest__scoring_carteira"):

    modelo_final.fit(X_history_final, y_history_final)

    # Score usado principalmente para ranking e priorização.
    score_churn = modelo_final.predict_proba(X_inference_final)[:, 1]

    mlflow.set_tags({
        "etapa": "modelo_final_scoring",
        "modelo_escolhido": MODELO_ESCOLHIDO,
        "conjunto_features": "modelo_enxuto",
        "uso": "geracao_da_carteira_priorizada",
    })

    mlflow.log_params({
        "modelo": "random_forest",
        "n_features": len(features_modelo_enxuto),
        "features": ", ".join(features_modelo_enxuto),
        "n_estimators": 400,
        "max_depth": 10,
        "min_samples_leaf": 20,
        "class_weight": "balanced",
        "seed": SEED,
    })

    mlflow.log_metrics({
        "clientes_treino_final": len(X_history_final),
        "clientes_score_carteira": len(X_inference_final),
        "score_medio_carteira": float(score_churn.mean()),
    })

    # Salva o pipeline completo no artefato da run do MLflow.
    mlflow.sklearn.log_model(
        sk_model=modelo_final,
        name="modelo_final_random_forest",
    )

print("Treino final e scoring concluídos.")

# COMMAND ----------

# ============================================================
# RANKING, FAIXAS DE PRIORIDADE E AÇÕES SUGERIDAS
# ============================================================

carteira_risco = inference_final[
    [
        "customerID",
        "tenure",
        "Contract",
        "PaymentMethod",
        "InternetService",
        "TechSupport",
        "MonthlyCharges",
        "TotalCharges",
        "MonthlyIncome",
    ]
].copy()

carteira_risco["score_churn"] = score_churn

# Ranking 1 = maior score de risco.
carteira_risco["ranking_risco"] = (
    carteira_risco["score_churn"]
    .rank(method="first", ascending=False)
    .astype(int)
)

total_clientes = len(carteira_risco)
limite_top_10 = int(np.ceil(total_clientes * 0.10))
limite_top_20 = int(np.ceil(total_clientes * 0.20))
limite_top_30 = int(np.ceil(total_clientes * 0.30))

carteira_risco["no_top_10"] = carteira_risco["ranking_risco"] <= limite_top_10
carteira_risco["no_top_20"] = carteira_risco["ranking_risco"] <= limite_top_20
carteira_risco["no_top_30"] = carteira_risco["ranking_risco"] <= limite_top_30

# Faixas exclusivas, mais fáceis de usar no dashboard.
carteira_risco["top_faixa"] = np.select(
    [
        carteira_risco["no_top_10"],
        carteira_risco["no_top_20"],
        carteira_risco["no_top_30"],
    ],
    [
        "Top 10%",
        "Top 10% a 20%",
        "Top 20% a 30%",
    ],
    default="Fora do Top 30%",
)

carteira_risco["faixa_risco"] = np.select(
    [
        carteira_risco["no_top_10"],
        carteira_risco["no_top_30"],
    ],
    [
        "Alto",
        "Medio",
    ],
    default="Baixo",
)

# Segmentação operacional vinda do estudo de retorno.
mensal = carteira_risco["Contract"].eq("Month-to-month")
fibra = carteira_risco["InternetService"].eq("Fiber optic")

condicoes_segmento = [
    carteira_risco["InternetService"].eq("DSL") & carteira_risco["tenure"].le(6),
    fibra & mensal & carteira_risco["TechSupport"].eq("No"),
    mensal & carteira_risco["PaymentMethod"].eq("Electronic check"),
    mensal & carteira_risco["tenure"].le(6),
    fibra & mensal,
]

nomes_segmento = [
    "DSL recente ate 6 meses",
    "Fibra mensal sem suporte",
    "Mensal com electronic check",
    "Recente mensal ate 6 meses",
    "Fibra mensal demais",
]

carteira_risco["segmento_acao"] = np.select(
    condicoes_segmento,
    nomes_segmento,
    default="Outros",
)

acoes = {
    "DSL recente ate 6 meses": (
        "Acompanhamento tecnico nos primeiros meses e avaliacao de migracao para fibra."
    ),
    "Fibra mensal sem suporte": (
        "Oferecer suporte tecnico temporario e avaliar incentivo para contrato anual."
    ),
    "Mensal com electronic check": (
        "Oferecer incentivo para migracao a debito automatico ou cartao."
    ),
    "Recente mensal ate 6 meses": (
        "Realizar onboarding proativo e avaliar incentivo para contrato anual."
    ),
    "Fibra mensal demais": (
        "Avaliar oferta de retencao e incentivo para contrato anual."
    ),
    "Outros": (
        "Monitorar perfil e definir abordagem conforme score, elegibilidade e capacidade."
    ),
}

carteira_risco["acao_sugerida"] = carteira_risco["segmento_acao"].map(acoes)

# Campos explícitos para futuro uso operacional.
carteira_risco["elegivel_contato"] = True
carteira_risco["data_geracao"] = pd.Timestamp.now()

carteira_risco = carteira_risco.sort_values(
    "ranking_risco"
).reset_index(drop=True)

display(carteira_risco.head(10))

# COMMAND ----------

display(carteira_risco.head(50))

# COMMAND ----------

# ============================================================
# CHECAGENS E GRAVAÇÃO NO UNITY CATALOG
# ============================================================

resumo_carteira = (
    carteira_risco.groupby("top_faixa", dropna=False)
    .agg(
        clientes=("customerID", "count"),
        score_medio=("score_churn", "mean"),
        score_minimo=("score_churn", "min"),
        mensalidade_media=("MonthlyCharges", "mean"),
    )
    .reindex(["Top 10%", "Top 10% a 20%", "Top 20% a 30%", "Fora do Top 30%"])
    .reset_index()
)

display(resumo_carteira.round(3))

assert len(carteira_risco) == len(inference_final)
assert carteira_risco["customerID"].is_unique
assert carteira_risco["score_churn"].between(0, 1).all()
assert carteira_risco["ranking_risco"].min() == 1
assert carteira_risco["ranking_risco"].max() == len(carteira_risco)

(
    spark.createDataFrame(carteira_risco)
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABELA_SAIDA)
)

print(f"Tabela criada: {TABELA_SAIDA}")
print(f"Top 10%: {limite_top_10} clientes")
print(f"Top 20%: {limite_top_20} clientes")
print(f"Top 30%: {limite_top_30} clientes")

# COMMAND ----------

carteira = spark.table("workspace.default.clientes_risco_churn").toPandas()

resumo_segmentos = (
    carteira.groupby("segmento_acao", dropna=False)
    .agg(
        clientes=("customerID", "count"),
        pct_carteira=("customerID", lambda x: len(x) / len(carteira)),
        score_medio=("score_churn", "mean"),
        score_mediano=("score_churn", "median"),
        score_maximo=("score_churn", "max"),
        clientes_top_10=("no_top_10", "sum"),
        clientes_top_20=("no_top_20", "sum"),
        clientes_top_30=("no_top_30", "sum"),
    )
    .reset_index()
)

for faixa, coluna in [
    ("top_10", "clientes_top_10"),
    ("top_20", "clientes_top_20"),
    ("top_30", "clientes_top_30"),
]:
    resumo_segmentos[f"pct_do_segmento_no_{faixa}"] = (
        resumo_segmentos[coluna] / resumo_segmentos["clientes"]
    )

    resumo_segmentos[f"pct_da_faixa_{faixa}"] = (
        resumo_segmentos[coluna] / resumo_segmentos[coluna].sum()
    )

resumo_segmentos = resumo_segmentos.sort_values(
    ["score_medio", "clientes"],
    ascending=False,
)

display(resumo_segmentos.round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC O modelo concentra as ações nos perfis em que o risco é mais elevado. Embora clientes de fibra mensal sem suporte representem 25% da carteira, eles concentram 97% do Top 10% de risco. Clientes DSL recentes aparecem como segundo público prioritário, sobretudo entre Top 20% e Top 30%.<br>
# MAGIC
# MAGIC | Evidência do EDA | O que o modelo encontrou na carteira | Interpretação |
# MAGIC |---|---|---|
# MAGIC | Churn maior nos primeiros 6 meses | 72,5% dos clientes **DSL recentes** estão no Top 30% | Tempo de casa baixo é um sinal relevante de fragilidade inicial |
# MAGIC | Contrato mensal está associado a maior churn, mesmo controlando tenure | Todos os segmentos prioritários usam contrato mensal | Falta de vínculo contratual é um risco consistente |
# MAGIC | Fibra tem maior churn que DSL | 97,2% do Top 10% está em **Fibra mensal sem suporte** | Fibra é o principal foco de risco, sobretudo em combinação com outros fatores |
# MAGIC | Falta de serviços agregados, especialmente suporte, está ligada a churn | 85,7% de “Fibra mensal sem suporte” entra no Top 30% | Suporte técnico é um ponto acionável, não apenas uma variável explicativa |
# MAGIC | Faturas maiores têm maior churn | Os clientes líderes do ranking têm cobranças mensais altas, em geral entre US$ 70 e US$ 100 | Reforça a hipótese de percepção de baixo valor pelo preço pago |
# MAGIC | Electronic check/manual está associado a maior churn | Electronic check sozinho só aparece mais no Top 20/30% | É sinal complementar; não justifica campanha cara sozinho |
# MAGIC
# MAGIC
# MAGIC O risco não está concentrado em uma variável isolada. Ele aumenta quando pouco tempo de casa, contrato mensal, fibra, falta de suporte e cobrança alta aparecem juntos.
# MAGIC
# MAGIC
# MAGIC
# MAGIC Quanto ao DSL<br>
# MAGIC O EDA já indicava que clientes novos tinham maior risco. O modelo refinou essa leitura:
# MAGIC - DSL recente não é tão extremo quanto fibra mensal sem suporte;
# MAGIC - mas tem score médio alto, 0,653;
# MAGIC - 38 dos 102 clientes entram no Top 20%;
# MAGIC - 74 dos 102 entram no Top 30%.
# MAGIC Então, DSL recente merece uma ação específica, mas não a ação mais cara da carteira.
# MAGIC
# MAGIC Uma leitura adequada seria:
# MAGIC - Fibra mensal sem suporte: prioridade de retenção intensiva;
# MAGIC - DSL recente: prioridade de onboarding técnico e prevenção nos primeiros meses;
# MAGIC - Electronic check mensal: incentivo mais escalável de migração de pagamento, principalmente se o cliente também estiver no Top 30%.
# MAGIC
# MAGIC
# MAGIC Comparando com um olhar para a análise de feedbacks,
# MAGIC
# MAGIC | Perfil de risco | Evidência do EDA/modelo | Feedback que ajuda a explicar | Ação coerente |
# MAGIC |---|---|---|---|
# MAGIC | Fibra mensal sem suporte | Principal concentração de risco | Concorrência, preço, qualidade/velocidade | Oferta de retenção + Diagnóstico técnico + suporte temporário + |
# MAGIC | DSL recente | Risco concentrado no Top 20/30% | Qualidade, atendimento e início de relacionamento | Onboarding técnico nos primeiros meses e contato preventivo |
# MAGIC | Mensal com electronic check | Risco intermediário | Preço/oferta melhor e baixa vinculação | Incentivo a pagamento automático, eventualmente associado a benefício |
# MAGIC | Contrato mensal | Presente em todos os públicos prioritários | Contrato/fidelidade e concorrência | Oferta de migração para vínculo anual, sem aplicar indiscriminadamente |
# MAGIC
# MAGIC
# MAGIC O EDA identificou que churn se associa a baixo vínculo, fibra, ausência de suporte, cobrança alta e pagamento manual. O modelo confirmou que essas variáveis, combinadas, concentram o risco: clientes de fibra, contrato mensal e sem suporte representam 25% da carteira, mas 97% do Top 10% de risco. Os feedbacks ajudam a transformar essa priorização em ações, apontando concorrência, contrato, qualidade e preço como os principais motivos declarados de saída.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Testando a importância das Variáveis do modelo final

# COMMAND ----------

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES_FINAIS = [
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "MonthlyIncome",
]

SEED = 42
TABELA_HISTORY = "workspace.default.history_modelagem"

history_shap = spark.table(TABELA_HISTORY).toPandas()

X_final = history_shap[FEATURES_FINAIS].copy()
y_final = history_shap["Churn"].astype(int).copy()

numericas = X_final.select_dtypes(include=np.number).columns.tolist()
categoricas = [c for c in FEATURES_FINAIS if c not in numericas]

preprocessador_final = ColumnTransformer(
    transformers=[
        ("numericas", SimpleImputer(strategy="median"), numericas),
        (
            "categoricas",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categoricas,
        ),
    ],
    remainder="drop",
)

modelo_rf_final = Pipeline([
    ("preprocessamento", preprocessador_final),
    ("modelo", RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )),
])

modelo_rf_final.fit(X_final, y_final)

print("Modelo final preparado para SHAP.")

# COMMAND ----------

# MAGIC %pip install shap

# COMMAND ----------

import shap
import matplotlib.pyplot as plt

# Amostra apenas para explicabilidade global.
# Não altera o modelo, as métricas, os scores nem as tabelas.
N_AMOSTRA_SHAP = min(500, len(X_final))
X_shap = X_final.sample(n=N_AMOSTRA_SHAP, random_state=SEED)

preprocessador = modelo_rf_final.named_steps["preprocessamento"]
floresta = modelo_rf_final.named_steps["modelo"]

X_shap_transformado = preprocessador.transform(X_shap)

# SHAP trabalha melhor com matriz densa nesta base pequena.
if hasattr(X_shap_transformado, "toarray"):
    X_shap_transformado = X_shap_transformado.toarray()

nomes_transformados = preprocessador.get_feature_names_out()

explainer = shap.TreeExplainer(floresta)
valores_shap = explainer.shap_values(X_shap_transformado)

# Compatível com versões do SHAP que retornam lista ou array 3D.
if isinstance(valores_shap, list):
    valores_churn = valores_shap[1]
elif valores_shap.ndim == 3:
    valores_churn = valores_shap[:, :, 1]
else:
    valores_churn = valores_shap

def feature_original(nome_transformado):
    for feature in FEATURES_FINAIS:
        if (
            nome_transformado == f"numericas__{feature}"
            or nome_transformado.startswith(f"categoricas__{feature}_")
        ):
            return feature
    return nome_transformado

importancia_detalhada = pd.DataFrame({
    "feature_transformada": nomes_transformados,
    "feature": [feature_original(nome) for nome in nomes_transformados],
    "importancia_shap": np.abs(valores_churn).mean(axis=0),
})

importancia_global = (
    importancia_detalhada
    .groupby("feature", as_index=False)["importancia_shap"]
    .sum()
    .sort_values("importancia_shap", ascending=False)
    .reset_index(drop=True)
)

display(importancia_global)

# COMMAND ----------

plt.figure(figsize=(10, 6))

plt.barh(
    importancia_global["feature"].iloc[::-1],
    importancia_global["importancia_shap"].iloc[::-1],
    color="#1F6FEB",
)

plt.title("Importância global das variáveis no score de churn")
plt.xlabel("Impacto médio absoluto no score (SHAP)")
plt.ylabel("")
plt.grid(axis="x", alpha=0.2)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC O SHAP mostra quais variáveis mais influenciam a priorização de churn do Random Forest. Ele explica o comportamento preditivo do modelo e não demonstra causalidade.

# COMMAND ----------

# Agrupa contribuições SHAP das dummies na feature original
mapeamento_features = pd.Series(
    [feature_original(nome) for nome in nomes_transformados],
    index=nomes_transformados,
)

shap_detalhado = pd.DataFrame(
    valores_churn,
    columns=nomes_transformados,
    index=X_shap.index,
)

shap_agrupado = pd.DataFrame(index=X_shap.index)

for feature in FEATURES_FINAIS:
    colunas_feature = mapeamento_features[
        mapeamento_features == feature
    ].index.tolist()

    shap_agrupado[feature] = shap_detalhado[colunas_feature].sum(axis=1)

importancia_global_corrigida = (
    shap_agrupado
    .abs()
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

importancia_global_corrigida.columns = ["feature", "importancia_shap"]

display(importancia_global_corrigida)

# COMMAND ----------

plt.figure(figsize=(10, 6))

plt.barh(
    importancia_global_corrigida["feature"].iloc[::-1],
    importancia_global_corrigida["importancia_shap"].iloc[::-1],
    color="#1F6FEB",
)

plt.title("Importância global das variáveis no score de churn")
plt.xlabel("Impacto médio absoluto no score (SHAP)")
plt.ylabel("")
plt.grid(axis="x", alpha=0.2)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Registrar evidências no MLflow
# MAGIC
# MAGIC Cada modelo tem métricas, parâmetros e gráficos de avaliação registrados.
# MAGIC Caso o workspace não permita criar o experimento compartilhado, altere
# MAGIC `EXPERIMENTO` para uma pasta onde a equipe tenha permissão de escrita.

# COMMAND ----------



# COMMAND ----------

mlflow.set_experiment(EXPERIMENTO)

for nome, pipeline in pipelines_ajustados.items():
    probabilidades = pipeline.predict_proba(X_test)[:, 1]
    metricas = metricas_teste(y_test, probabilidades)

    with mlflow.start_run(run_name=nome):
        mlflow.set_tags({
            "projeto": "terra_signal_churn",
            "cenario_pagamento": "PaymentMethod_completo",
            "features_sensiveis_incluidas": "nao",
            "target": "Churn",
        })
        mlflow.log_params({
            "modelo": nome,
            "n_features_antes_do_onehot": len(features_modelo),
            "seed": SEED,
            "test_size": 0.20,
            "cv_folds": 5,
        })
        mlflow.log_metrics({f"test_{chave}": float(valor) for chave, valor in metricas.items()})

        with tempfile.TemporaryDirectory() as diretorio:
            caminho_roc = os.path.join(diretorio, "curva_roc.png")
            caminho_pr = os.path.join(diretorio, "curva_precision_recall.png")

            fig, ax = plt.subplots(figsize=(6, 5))
            RocCurveDisplay.from_predictions(y_test, probabilidades, ax=ax, name=nome)
            ax.set_title(f"ROC — {nome}")
            fig.tight_layout()
            fig.savefig(caminho_roc, dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6, 5))
            PrecisionRecallDisplay.from_predictions(y_test, probabilidades, ax=ax, name=nome)
            ax.set_title(f"Precision-Recall — {nome}")
            fig.tight_layout()
            fig.savefig(caminho_pr, dpi=150)
            plt.close(fig)

            mlflow.log_artifact(caminho_roc)
            mlflow.log_artifact(caminho_pr)
            mlflow.sklearn.log_model(pipeline, name="model")

print("Runs registrados no MLflow.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Criando tabelas para alimentar cen[arios de decisäo - preven;áo e urgëncia]
# MAGIC

# COMMAND ----------

# ============================================================
# CENÁRIOS FINANCEIROS DE RETENÇÃO
# 3 custos x 3 taxas de contato/adesão x 3 efetividades = 27
# ============================================================

import numpy as np
import pandas as pd
from itertools import product

CATALOG = "workspace.default"
TABELA_CARTEIRA = f"{CATALOG}.clientes_risco_churn"
TABELA_CENARIOS = f"{CATALOG}.cenarios_retorno_churn"

HORIZONTES_MESES = [12, 18, 24]
MARGEM_EBITDA = 0.35

# Taxas observadas no teste final do Random Forest.
# Elas são usadas como estimativa do churn esperado em cada público.
TAXA_CHURN_TESTE_POR_PUBLICO = {
    "Top 10%": 84 / 113,   # 74,3%
    "Top 20%": 154 / 226,  # 68,1%
    "Top 30%": 203 / 339,  # 59,9%
}

COLUNA_PUBLICO = {
    "Top 10%": "no_top_10",
    "Top 20%": "no_top_20",
    "Top 30%": "no_top_30",
}

ORDEM_PUBLICOS = {
    "Top 10%": 1,
    "Top 20%": 2,
    "Top 30%": 3,
}

# ------------------------------------------------------------
# 1. Matriz-base de custos por segmento e faixa de custo
# ------------------------------------------------------------
# Mantida para todos os segmentos e para clientes com mais de 6 meses.
# Ela também é a referência de comparação da política atual.

MATRIZ_CUSTOS = {
    "Fibra mensal sem suporte": {
        "Baixo": 80,
        "Intermediario": 120,
        "Alto": 180,
    },
    "DSL recente ate 6 meses": {
        "Baixo": 30,
        "Intermediario": 60,
        "Alto": 100,
    },
    "Mensal com electronic check": {
        "Baixo": 20,
        "Intermediario": 40,
        "Alto": 60,
    },
    "Fibra mensal demais": {
        "Baixo": 40,
        "Intermediario": 80,
        "Alto": 140,
    },
    "Recente mensal ate 6 meses": {
        "Baixo": 25,
        "Intermediario": 50,
        "Alto": 90,
    },
    "Outros": {
        "Baixo": 0,
        "Intermediario": 0,
        "Alto": 0,
    },
}

# Política complementar:
# clientes de Fibra mensal sem suporte com tenure até 6 meses.
#
# Baixo: comunicação digital + pagamento automático + contrato anual.
# Intermediário: contato proativo assistido + pequeno incentivo.
# Alto: atendimento remoto reforçado + benefício maior.
#
# Uma visita técnica real continua sendo exceção operacional e pode usar
# o custo alto-base de R$ 180, fora da regra padrão do cenário.

CUSTOS_FIBRA_ATE_6_MESES = {
    "Baixo": 30,
    "Intermediario": 60,
    "Alto": 120,
}

# A tabela-base permanece com as mesmas três colunas e 18 linhas.
# Isso ajuda a preservar a estrutura intermediária já esperada no notebook.
linhas_custo = []

for segmento, custos in MATRIZ_CUSTOS.items():
    for faixa_custo, custo_por_cliente in custos.items():
        linhas_custo.append({
            "segmento_acao": segmento,
            "faixa_custo": faixa_custo,
            "custo_por_cliente": float(custo_por_cliente),
        })

matriz_custos = pd.DataFrame(linhas_custo)

display(matriz_custos)

# Evidência explícita da regra nova, sem alterar a tabela-base.
display(
    pd.DataFrame(
        [
            {
                "segmento_acao": "Fibra mensal sem suporte",
                "faixa_tenure": "0–6 meses",
                "faixa_custo": faixa,
                "custo_por_cliente": float(custo),
            }
            for faixa, custo in CUSTOS_FIBRA_ATE_6_MESES.items()
        ]
    )
)

# COMMAND ----------

# ------------------------------------------------------------
# 2. Criar as 27 combinações de premissas por segmento
# ------------------------------------------------------------

TAXAS_CONTATO_ADESAO = [0.70, 0.80, 0.90]
EFETIVIDADES_RETENCAO = [0.10, 0.20, 0.30]

premissas_operacionais = pd.DataFrame(
    product(
        TAXAS_CONTATO_ADESAO,
        EFETIVIDADES_RETENCAO,
    ),
    columns=[
        "taxa_contato_adesao",
        "efetividade_retencao",
    ],
)

# 3 faixas de custo x 3 taxas de contato/adesão x 3 efetividades = 27.
premissas_27 = matriz_custos.merge(
    premissas_operacionais,
    how="cross",
)

assert len(premissas_27) == len(MATRIZ_CUSTOS) * 27

print(
    "Combinações por segmento:",
    premissas_27.groupby("segmento_acao").size().unique()
)

premissas_cenarios = premissas_27.merge(
    pd.DataFrame({"horizonte_meses": HORIZONTES_MESES}),
    how="cross",
)

assert len(premissas_cenarios) == len(MATRIZ_CUSTOS) * 27 * len(HORIZONTES_MESES)

display(premissas_cenarios.head())

# COMMAND ----------

# ------------------------------------------------------------
# 3. Base da carteira: público Top X% x segmento
# ------------------------------------------------------------

carteira = spark.table(TABELA_CARTEIRA).toPandas()

assert carteira["customerID"].is_unique
assert carteira["score_churn"].between(0, 1).all()

# Hoje todos estão elegíveis por padrão.
# Futuramente, a regra pode excluir clientes sem consentimento,
# em campanha recente ou já em processo de cancelamento.
carteira_elegivel = carteira[
    carteira["elegivel_contato"].eq(True)
].copy()

bases_publico = []

for publico_alvo, coluna_publico in COLUNA_PUBLICO.items():

    base_publico = carteira_elegivel[
        carteira_elegivel[coluna_publico].eq(True)
    ].copy()

    # Grupo elegível para a política de custo personalizada.
    base_publico["cliente_fibra_ate_6_meses"] = (
        base_publico["segmento_acao"].eq("Fibra mensal sem suporte")
        & base_publico["tenure"].le(6)
    ).astype(int)

    resumo = (
        base_publico
        .groupby("segmento_acao", dropna=False)
        .agg(
            clientes_abordados=("customerID", "count"),
            mensalidade_media=("MonthlyCharges", "mean"),
            score_medio=("score_churn", "mean"),
            clientes_fibra_ate_6_meses=(
                "cliente_fibra_ate_6_meses",
                "sum",
            ),
        )
        .reset_index()
    )

    resumo["publico_alvo"] = publico_alvo
    resumo["ordem_publico"] = ORDEM_PUBLICOS[publico_alvo]
    resumo["taxa_churn_estimada"] = TAXA_CHURN_TESTE_POR_PUBLICO[publico_alvo]

    resumo["churners_esperados"] = (
        resumo["clientes_abordados"]
        * resumo["taxa_churn_estimada"]
    )

    bases_publico.append(resumo)

base_cenarios = pd.concat(bases_publico, ignore_index=True)

# Checagens da regra personalizada.
assert (
    base_cenarios["clientes_fibra_ate_6_meses"]
    <= base_cenarios["clientes_abordados"]
).all()

assert (
    base_cenarios.loc[
        base_cenarios["segmento_acao"].ne("Fibra mensal sem suporte"),
        "clientes_fibra_ate_6_meses",
    ]
    .eq(0)
    .all()
)

display(
    base_cenarios
    .sort_values(["ordem_publico", "score_medio"], ascending=[True, False])
    .round(3)
)

# COMMAND ----------

# ------------------------------------------------------------
# 4. Cruzar público/segmento com premissas de custo,
# adesão, retenção e horizonte
# ------------------------------------------------------------

cenarios = base_cenarios.merge(
    premissas_cenarios,
    on="segmento_acao",
    how="left",
    validate="many_to_many",
)

# ------------------------------------------------------------
# Personalização de custo para Fibra mensal sem suporte até 6 meses
#
# Mantém uma única linha por público + segmento + cenário.
# O custo_por_cliente passa a ser a média ponderada do segmento:
#
# custo médio =
#   (clientes com mais de 6 meses × custo-base
#    + clientes até 6 meses × custo personalizado)
#   / clientes abordados
#
# Como a mesma taxa de adesão é aplicada aos dois subgrupos,
# usar o custo médio preserva exatamente o custo total esperado.
# ------------------------------------------------------------

cenarios["custo_base_por_cliente"] = cenarios["custo_por_cliente"]

mascara_fibra_personalizada = (
    cenarios["segmento_acao"].eq("Fibra mensal sem suporte")
)

custo_fibra_ate_6 = cenarios["faixa_custo"].map(
    CUSTOS_FIBRA_ATE_6_MESES
)

assert custo_fibra_ate_6[mascara_fibra_personalizada].notna().all(), (
    "Há faixa de custo de Fibra sem valor definido "
    "em CUSTOS_FIBRA_ATE_6_MESES."
)

cenarios.loc[mascara_fibra_personalizada, "custo_por_cliente"] = (
    (
        (
            cenarios.loc[
                mascara_fibra_personalizada,
                "clientes_abordados",
            ]
            - cenarios.loc[
                mascara_fibra_personalizada,
                "clientes_fibra_ate_6_meses",
            ]
        )
        * cenarios.loc[
            mascara_fibra_personalizada,
            "custo_base_por_cliente",
        ]
        + (
            cenarios.loc[
                mascara_fibra_personalizada,
                "clientes_fibra_ate_6_meses",
            ]
            * custo_fibra_ate_6[mascara_fibra_personalizada]
        )
    )
    / cenarios.loc[
        mascara_fibra_personalizada,
        "clientes_abordados",
    ]
)

# Checagens: os demais segmentos não podem ter custo alterado.
assert (
    cenarios.loc[
        ~mascara_fibra_personalizada,
        "custo_por_cliente",
    ]
    .eq(
        cenarios.loc[
            ~mascara_fibra_personalizada,
            "custo_base_por_cliente",
        ]
    )
    .all()
)

assert cenarios["custo_por_cliente"].notna().all()

# Evidência da personalização: uma linha por público e faixa de custo.
validacao_custos_personalizados = (
    cenarios[
        (cenarios["taxa_contato_adesao"] == 0.80)
        & (cenarios["efetividade_retencao"] == 0.30)
        & (cenarios["horizonte_meses"] == 24)
        & mascara_fibra_personalizada
    ][
        [
            "publico_alvo",
            "faixa_custo",
            "clientes_abordados",
            "clientes_fibra_ate_6_meses",
            "custo_base_por_cliente",
            "custo_por_cliente",
        ]
    ]
    .drop_duplicates()
    .sort_values(["publico_alvo", "faixa_custo"])
)

display(validacao_custos_personalizados.round(2))

assert cenarios["custo_por_cliente"].notna().all(), (
    "Há segmento sem custo definido na MATRIZ_CUSTOS."
)

# Custo variável: ocorre somente para clientes que aderem à oferta/ação.
cenarios["custo_total"] = (
    cenarios["clientes_abordados"]
    * cenarios["taxa_contato_adesao"]
    * cenarios["custo_por_cliente"]
)

# Clientes que aderem à oferta, sob a premissa do cenário.
cenarios["clientes_contatados_ou_aderentes_estimados"] = (
    cenarios["clientes_abordados"]
    * cenarios["taxa_contato_adesao"]
)

# Churners potencialmente retidos.
cenarios["churners_retidos_potencial"] = (
    cenarios["churners_esperados"]
    * cenarios["taxa_contato_adesao"]
    * cenarios["efetividade_retencao"]
)

# Receita e EBITDA em risco no horizonte selecionado.
cenarios["receita_em_risco_bruta_horizonte"] = (
    cenarios["churners_esperados"]
    * cenarios["mensalidade_media"]
    * cenarios["horizonte_meses"]
)

cenarios["ebitda_em_risco_horizonte"] = (
    cenarios["receita_em_risco_bruta_horizonte"]
    * MARGEM_EBITDA
)

# Receita potencialmente preservada no horizonte selecionado.
cenarios["receita_preservada_bruta_potencial"] = (
    cenarios["churners_retidos_potencial"]
    * cenarios["mensalidade_media"]
    * cenarios["horizonte_meses"]
)

# EBITDA potencialmente preservado: 35% da receita, conforme premissa da TS.
cenarios["ebitda_preservado_potencial"] = (
    cenarios["receita_preservada_bruta_potencial"]
    * MARGEM_EBITDA
)

# ------------------------------------------------------------
# Visão financeira: receita bruta
# ------------------------------------------------------------

cenarios["resultado_liquido_bruto_potencial"] = (
    cenarios["receita_preservada_bruta_potencial"]
    - cenarios["custo_total"]
)

cenarios["roi_bruto_potencial_pct"] = np.where(
    cenarios["custo_total"] > 0,
    100
    * cenarios["resultado_liquido_bruto_potencial"]
    / cenarios["custo_total"],
    np.nan,
)

# ------------------------------------------------------------
# Visão financeira: EBITDA — visão principal de decisão
# ------------------------------------------------------------

cenarios["resultado_liquido_ebitda_potencial"] = (
    cenarios["ebitda_preservado_potencial"]
    - cenarios["custo_total"]
)

cenarios["roi_ebitda_potencial_pct"] = np.where(
    cenarios["custo_total"] > 0,
    100
    * cenarios["resultado_liquido_ebitda_potencial"]
    / cenarios["custo_total"],
    np.nan,
)

# Mantém nomes genéricos apontando para a visão EBITDA.
cenarios["resultado_liquido_potencial"] = (
    cenarios["resultado_liquido_ebitda_potencial"]
)

cenarios["roi_potencial_pct"] = (
    cenarios["roi_ebitda_potencial_pct"]
)

cenarios["cenario"] = (
    "Custo " + cenarios["faixa_custo"]
    + " | adesao "
    + (cenarios["taxa_contato_adesao"] * 100).astype(int).astype(str) + "%"
    + " | retencao "
    + (cenarios["efetividade_retencao"] * 100).astype(int).astype(str) + "%"
    + " | horizonte "
    + cenarios["horizonte_meses"].astype(int).astype(str) + " meses"
)

cenarios["margem_ebitda_utilizada"] = MARGEM_EBITDA
cenarios["data_geracao"] = pd.Timestamp.now()

colunas_saida = [
    "cenario",
    "publico_alvo",
    "ordem_publico",
    "segmento_acao",
    "faixa_custo",
    "clientes_abordados",
    "clientes_contatados_ou_aderentes_estimados",
    "score_medio",
    "taxa_churn_estimada",
    "churners_esperados",
    "churners_retidos_potencial",
    "mensalidade_media",
    "receita_em_risco_bruta_horizonte",
    "ebitda_em_risco_horizonte",
    "custo_por_cliente",
    "custo_total",
    "taxa_contato_adesao",
    "efetividade_retencao",
    "receita_preservada_bruta_potencial",
    "ebitda_preservado_potencial",
    "resultado_liquido_bruto_potencial",
    "roi_bruto_potencial_pct",
    "resultado_liquido_ebitda_potencial",
    "roi_ebitda_potencial_pct",
    "resultado_liquido_potencial",
    "roi_potencial_pct",
    "horizonte_meses",
    "margem_ebitda_utilizada",
    "data_geracao",
]

cenarios = (
    cenarios[colunas_saida]
    .sort_values(
        [
            "ordem_publico",
            "horizonte_meses",
            "segmento_acao",
            "faixa_custo",
            "taxa_contato_adesao",
            "efetividade_retencao",
        ]
    )
    .reset_index(drop=True)
)

print(f"Total de linhas: {len(cenarios):,}")
display(cenarios.head(30).round(2))

# COMMAND ----------

# ------------------------------------------------------------
# Visão executiva agregada para o dashboard
# Um cenário por público + premissas, somando todos os segmentos.
# ------------------------------------------------------------

chaves_dashboard = [
    "cenario",
    "publico_alvo",
    "ordem_publico",
    "faixa_custo",
    "taxa_contato_adesao",
    "efetividade_retencao",
    "horizonte_meses",
    "margem_ebitda_utilizada",
]

cenarios_dashboard = (
    cenarios
    .groupby(chaves_dashboard, as_index=False)
    .agg(
        segmentos_incluidos=("segmento_acao", "nunique"),
        clientes_abordados=("clientes_abordados", "sum"),
        clientes_aderentes_estimados=(
            "clientes_contatados_ou_aderentes_estimados", "sum"
        ),
        churners_esperados=("churners_esperados", "sum"),
        churners_retidos_potencial=("churners_retidos_potencial", "sum"),
        receita_em_risco_bruta_horizonte=(
            "receita_em_risco_bruta_horizonte", "sum"
        ),
        ebitda_em_risco_horizonte=(
            "ebitda_em_risco_horizonte", "sum"
        ),
        custo_total=("custo_total", "sum"),
        receita_preservada_bruta_potencial=(
            "receita_preservada_bruta_potencial", "sum"
        ),
        ebitda_preservado_potencial=(
            "ebitda_preservado_potencial", "sum"
        ),
        resultado_liquido_bruto_potencial=(
            "resultado_liquido_bruto_potencial", "sum"
        ),
        resultado_liquido_ebitda_potencial=(
            "resultado_liquido_ebitda_potencial", "sum"
        ),
    )
)

# ROI calculado após somar os segmentos.
cenarios_dashboard["roi_bruto_potencial_pct"] = np.where(
    cenarios_dashboard["custo_total"] > 0,
    100
    * cenarios_dashboard["resultado_liquido_bruto_potencial"]
    / cenarios_dashboard["custo_total"],
    np.nan,
)

cenarios_dashboard["roi_ebitda_potencial_pct"] = np.where(
    cenarios_dashboard["custo_total"] > 0,
    100
    * cenarios_dashboard["resultado_liquido_ebitda_potencial"]
    / cenarios_dashboard["custo_total"],
    np.nan,
)

cenarios_dashboard["resultado_liquido_potencial"] = (
    cenarios_dashboard["resultado_liquido_ebitda_potencial"]
)

cenarios_dashboard["roi_potencial_pct"] = (
    cenarios_dashboard["roi_ebitda_potencial_pct"]
)

cenarios_dashboard = (
    cenarios_dashboard
    .sort_values([
        "ordem_publico",
        "horizonte_meses",
        "faixa_custo",
        "taxa_contato_adesao",
        "efetividade_retencao",
    ])
    .reset_index(drop=True)
)

# 3 públicos × 3 custos × 3 adesões × 3 efetividades × 3 horizontes
assert len(cenarios_dashboard) == 3 * 3 * 3 * 3 * 3  # 243 linhas

display(cenarios_dashboard.round(2))

# COMMAND ----------

# ------------------------------------------------------------
# 5. Visão agregada para dashboard
# Um cenário por público + premissas, somando segmentos.
# ------------------------------------------------------------

chaves_dashboard = [
    "cenario",
    "publico_alvo",
    "ordem_publico",
    "faixa_custo",
    "taxa_contato_adesao",
    "efetividade_retencao",
    "horizonte_meses",
    "margem_ebitda_utilizada",
]

cenarios_dashboard = (
    cenarios
    .groupby(chaves_dashboard, as_index=False)
    .agg(
        segmentos_incluidos=("segmento_acao", "nunique"),
        clientes_abordados=("clientes_abordados", "sum"),
        clientes_aderentes_estimados=(
            "clientes_contatados_ou_aderentes_estimados", "sum"
        ),
        churners_esperados=("churners_esperados", "sum"),
        churners_retidos_potencial=("churners_retidos_potencial", "sum"),
        receita_em_risco_bruta_horizonte=(
            "receita_em_risco_bruta_horizonte", "sum"
        ),
        ebitda_em_risco_horizonte=(
            "ebitda_em_risco_horizonte", "sum"
        ),
        custo_total=("custo_total", "sum"),
        receita_preservada_bruta_potencial=(
            "receita_preservada_bruta_potencial", "sum"
        ),
        ebitda_preservado_potencial=(
            "ebitda_preservado_potencial", "sum"
        ),
        resultado_liquido_bruto_potencial=(
            "resultado_liquido_bruto_potencial", "sum"
        ),
        resultado_liquido_ebitda_potencial=(
            "resultado_liquido_ebitda_potencial", "sum"
        ),
    )
)

# ROI calculado após a soma dos segmentos.
cenarios_dashboard["roi_bruto_potencial_pct"] = np.where(
    cenarios_dashboard["custo_total"] > 0,
    100
    * cenarios_dashboard["resultado_liquido_bruto_potencial"]
    / cenarios_dashboard["custo_total"],
    np.nan,
)

cenarios_dashboard["roi_ebitda_potencial_pct"] = np.where(
    cenarios_dashboard["custo_total"] > 0,
    100
    * cenarios_dashboard["resultado_liquido_ebitda_potencial"]
    / cenarios_dashboard["custo_total"],
    np.nan,
)

cenarios_dashboard["resultado_liquido_potencial"] = (
    cenarios_dashboard["resultado_liquido_ebitda_potencial"]
)

cenarios_dashboard["roi_potencial_pct"] = (
    cenarios_dashboard["roi_ebitda_potencial_pct"]
)

assert len(cenarios_dashboard) == 243, (
    f"Esperado: 243 cenários; encontrado: {len(cenarios_dashboard)}"
)

cenarios_dashboard = (
    cenarios_dashboard
    .sort_values(
        [
            "ordem_publico",
            "horizonte_meses",
            "faixa_custo",
            "taxa_contato_adesao",
            "efetividade_retencao",
        ]
    )
    .reset_index(drop=True)
)

print(f"Total de cenários agregados: {len(cenarios_dashboard):,}")
display(cenarios_dashboard.head(30).round(2))

# COMMAND ----------

cenarios_dashboard[
    (cenarios_dashboard["taxa_contato_adesao"] == 0.80)
    & (cenarios_dashboard["efetividade_retencao"] == 0.30)
    & (cenarios_dashboard["horizonte_meses"] == 12)
][
    [
        "publico_alvo",
        "faixa_custo",
        "custo_total",
        "ebitda_preservado_potencial",
        "resultado_liquido_ebitda_potencial",
        "roi_ebitda_potencial_pct",
    ]
].sort_values(["publico_alvo", "faixa_custo"])

# COMMAND ----------

colunas_leitura = [
    "publico_alvo",
    "horizonte_meses",
    "faixa_custo",
    "taxa_contato_adesao",
    "efetividade_retencao",
    "clientes_abordados",
    "clientes_aderentes_estimados",
    "custo_total",
    "churners_retidos_potencial",
    "ebitda_preservado_potencial",
    "resultado_liquido_ebitda_potencial",
    "roi_ebitda_potencial_pct",
]

cenarios_positivos = (
    cenarios_dashboard[
        cenarios_dashboard["roi_ebitda_potencial_pct"] > 0
    ]
    .sort_values(
        [
            "horizonte_meses",
            "roi_ebitda_potencial_pct",
        ],
        ascending=[True, False],
    )
    .reset_index(drop=True)
)

print(f"Cenários com ROI EBITDA positivo: {len(cenarios_positivos)}")
display(cenarios_positivos[colunas_leitura].round(2))

# COMMAND ----------

# ------------------------------------------------------------
# 5. Checagens e gravação no Unity Catalog
# ------------------------------------------------------------

quantidade_por_publico_segmento = (
    cenarios
    .groupby(["publico_alvo", "segmento_acao"])
    .size()
    .reset_index(name="quantidade_cenarios")
)

display(quantidade_por_publico_segmento)

ESPERADO_POR_PUBLICO_SEGMENTO = 27 * len(HORIZONTES_MESES)

ESPERADO_POR_PUBLICO_SEGMENTO = 27 * len(HORIZONTES_MESES)

assert quantidade_por_publico_segmento["quantidade_cenarios"].eq(
    ESPERADO_POR_PUBLICO_SEGMENTO
).all(), (
    f"Cada combinação público + segmento deveria ter "
    f"{ESPERADO_POR_PUBLICO_SEGMENTO} cenários."
)

# No estado atual:
# 12 combinações público + segmento × 81 cenários cada = 972 linhas.
assert len(cenarios) == 972, (
    f"Esperado: 972 linhas na tabela detalhada; "
    f"encontrado: {len(cenarios)}."
)

# Cria o DataFrame Spark antes de gravar para validar o schema.
cenarios_spark = spark.createDataFrame(cenarios)

# Garante que as colunas e os tipos permaneçam iguais aos da tabela atual.
schema_atual = spark.table(TABELA_CENARIOS).schema
schema_novo = cenarios_spark.schema

colunas_atuais = [campo.name for campo in schema_atual]
colunas_novas = [campo.name for campo in schema_novo]

assert colunas_novas == colunas_atuais, (
    "As colunas da nova tabela diferem da estrutura atual. "
    f"Atual: {colunas_atuais} | Nova: {colunas_novas}"
)

tipos_atuais = [
    (campo.name, campo.dataType.simpleString())
    for campo in schema_atual
]

tipos_novos = [
    (campo.name, campo.dataType.simpleString())
    for campo in schema_novo
]

assert tipos_novos == tipos_atuais, (
    "Os tipos de dados da nova tabela diferem da estrutura atual. "
    f"Atual: {tipos_atuais} | Nova: {tipos_novos}"
)

# Só executa se todas as validações acima passarem.
(
    cenarios_spark
    .write
    .mode("overwrite")
    .saveAsTable(TABELA_CENARIOS)
)

print(f"Tabela detalhada atualizada: {TABELA_CENARIOS}")
print(f"Linhas gravadas: {len(cenarios):,}")

# COMMAND ----------

TABELA_CENARIOS_DASHBOARD = (
    f"{CATALOG}.cenarios_retorno_churn_dashboard"
)

assert len(cenarios_dashboard) == 243, (
    f"Esperado: 243 linhas na tabela agregada; "
    f"encontrado: {len(cenarios_dashboard)}."
)

cenarios_dashboard_spark = spark.createDataFrame(cenarios_dashboard)

schema_atual_dashboard = spark.table(
    TABELA_CENARIOS_DASHBOARD
).schema

schema_novo_dashboard = cenarios_dashboard_spark.schema

colunas_atuais_dashboard = [
    campo.name for campo in schema_atual_dashboard
]

colunas_novas_dashboard = [
    campo.name for campo in schema_novo_dashboard
]

assert colunas_novas_dashboard == colunas_atuais_dashboard, (
    "As colunas da nova tabela agregada diferem da estrutura atual. "
    f"Atual: {colunas_atuais_dashboard} | "
    f"Nova: {colunas_novas_dashboard}"
)

tipos_atuais_dashboard = [
    (campo.name, campo.dataType.simpleString())
    for campo in schema_atual_dashboard
]

tipos_novos_dashboard = [
    (campo.name, campo.dataType.simpleString())
    for campo in schema_novo_dashboard
]

assert tipos_novos_dashboard == tipos_atuais_dashboard, (
    "Os tipos de dados da nova tabela agregada diferem da estrutura atual. "
    f"Atual: {tipos_atuais_dashboard} | "
    f"Nova: {tipos_novos_dashboard}"
)

(
    cenarios_dashboard_spark
    .write
    .mode("overwrite")
    .saveAsTable(TABELA_CENARIOS_DASHBOARD)
)

print(
    f"Tabela agregada atualizada: "
    f"{TABELA_CENARIOS_DASHBOARD}"
)
print(f"Linhas gravadas: {len(cenarios_dashboard):,}")

# COMMAND ----------

display(
    spark.table("workspace.default.clientes_risco_churn")
)

# COMMAND ----------

from datetime import datetime
from pyspark.sql import functions as F

NOME_EQUIPE = "equipe05"

predicao = (
    spark.table("workspace.default.clientes_risco_churn")
    .select(
        "customerID",
        F.when(F.col("score_churn") >= 0.5, F.lit("Yes"))
         .otherwise(F.lit("No"))
         .alias("prediction")
    )
)

# Validações antes de exportar
assert predicao.count() == 1409, "Quantidade de linhas diferente da base de inferência."
assert predicao.select("customerID").distinct().count() == 1409, "customerID duplicado."
assert predicao.filter(~F.col("prediction").isin("Yes", "No")).count() == 0

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
arquivo = f"prediction_{NOME_EQUIPE}_{timestamp}.csv"

# A base é pequena; exporta um único CSV sem índice extra.
predicao.toPandas().to_csv(arquivo, index=False)

print(f"Arquivo gerado: {arquivo}")
display(predicao.limit(10))

# COMMAND ----------

