"""Classificação do CustomerFeedback com a laya e aplicação da revisão.

Gera o feedback_laya_revisado.csv, que a análise exploratória (Ato 5) lê. Três etapas:

    python laya_feedback.py classificar   # 1. a laya classifica todos os feedbacks -> feedback_laya_full.csv
    python laya_feedback.py lotes         # 2. divide a classificação em lotes de 200 para revisão -> revisao/lote_XX.csv
    python laya_feedback.py juntar        # 3. aplica as correções da revisão -> feedback_laya_revisado.csv

Entre as etapas 2 e 3, cada lote é revisado seguindo revisao/INSTRUCOES.md, e só as classificações erradas
vão para revisao/correcoes_XX.csv (colunas: customerID, atrito_corrigido, justificativa).

Ressalva: a revisão viu a coluna Churn. Isso não atrapalha o uso desta base para entender POR QUE os
clientes saem, mas a torna inadequada como variável para PREVER o churn (vazamento do alvo).
"""
import sys
from pathlib import Path

import pandas as pd

PASTA = Path(__file__).parent
ARQUIVO_HISTORY = PASTA / "history.csv"
ARQUIVO_LAYA = PASTA / "feedback_laya_full.csv"
ARQUIVO_REVISADO = PASTA / "feedback_laya_revisado.csv"
PASTA_REVISAO = PASTA / "revisao"

MODELO_LAYA = "convaiinnovations/laya"   # checkpoint em inglês, usado sem treino específico (zero-shot)
TAMANHO_LOTE_LAYA = 8                    # textos por chamada à laya
TAMANHO_LOTE_REVISAO = 200               # linhas por arquivo de revisão

# Uma única pergunta, válida para quem saiu e para quem ficou: qual é o principal ponto de atrito?
PERGUNTA = {
    "atrito": {
        "type": "choice",
        "instructions": "What is the customer's main complaint or concern about the company?",
        "criteria": {
            "price": "monthly charges are too high, expensive, poor value for the money",
            "service_quality": "internet is slow or unreliable, connection problems, frequent outages",
            "competitor": "another provider offers a better deal or package",
            "payment": "the payment method is inconvenient, payment issues or billing delays",
            "personal": "personal reasons not related to the service, such as moving or a change in life",
            "satisfied": "the customer is satisfied and has no relevant complaint",
        },
    },
}
CATEGORIAS = list(PERGUNTA["atrito"]["criteria"])


# ==============================================================================================
# Etapa 1: classificação com a laya
# ==============================================================================================
def classificar():
    base = pd.read_csv(ARQUIVO_HISTORY, usecols=["customerID", "CustomerFeedback", "Churn"])

    # Retoma de onde parou: cada lote é salvo assim que termina (no CPU, a base inteira leva cerca de 1 hora)
    feitos = pd.read_csv(ARQUIVO_LAYA)["customerID"] if ARQUIVO_LAYA.exists() else pd.Series(dtype=str)
    faltam = base[~base["customerID"].isin(feitos)]
    print(f"Total: {len(base)} | já classificados: {len(feitos)} | faltam: {len(faltam)}")
    if faltam.empty:
        return

    import laya   # pip install laya

    agente = laya.Agent(MODELO_LAYA)
    for inicio in range(0, len(faltam), TAMANHO_LOTE_LAYA):
        lote = faltam.iloc[inicio:inicio + TAMANHO_LOTE_LAYA]
        respostas = agente.predict_batch(lote["CustomerFeedback"].tolist(), PERGUNTA)

        linhas = []
        for cliente, churn, resposta in zip(lote["customerID"], lote["Churn"], respostas):
            r = resposta["answers"]["atrito"]
            linhas.append({
                "customerID": cliente,
                "Churn": churn,
                "atrito": r["choice"],
                "confianca": r["answer_confidence"],
                **{f"p_{k}": v for k, v in r["probabilities"].items()},
            })
        pd.DataFrame(linhas).to_csv(ARQUIVO_LAYA, mode="a", header=not ARQUIVO_LAYA.exists(), index=False)
        print(f"{inicio + len(lote)}/{len(faltam)}")

    print("Classificação concluída.")


# ==============================================================================================
# Etapa 2: lotes para revisão
# ==============================================================================================
def lotes():
    laya = pd.read_csv(ARQUIVO_LAYA).drop_duplicates("customerID")
    textos = pd.read_csv(ARQUIVO_HISTORY, usecols=["customerID", "CustomerFeedback"])
    base = textos.merge(laya[["customerID", "Churn", "atrito", "confianca"]], on="customerID", how="inner")

    PASTA_REVISAO.mkdir(exist_ok=True)
    inicios = range(0, len(base), TAMANHO_LOTE_REVISAO)
    for numero, inicio in enumerate(inicios, start=1):
        base.iloc[inicio:inicio + TAMANHO_LOTE_REVISAO].to_csv(PASTA_REVISAO / f"lote_{numero:02d}.csv", index=False)
    print(f"{len(base)} linhas -> {len(inicios)} lotes em {PASTA_REVISAO}")


# ==============================================================================================
# Etapa 3: aplicação das correções
# ==============================================================================================
def juntar():
    laya = pd.read_csv(ARQUIVO_LAYA).drop_duplicates("customerID")

    arquivos = sorted(PASTA_REVISAO.glob("correcoes_*.csv"))
    correcoes = pd.concat([pd.read_csv(arquivo) for arquivo in arquivos], ignore_index=True)

    invalidas = correcoes[~correcoes["atrito_corrigido"].isin(CATEGORIAS)]
    if len(invalidas):
        raise SystemExit(f"Categorias inválidas nas correções:\n{invalidas}")
    correcoes = correcoes.drop_duplicates("customerID", keep="last")

    revisado = laya.rename(columns={"atrito": "atrito_laya"}).merge(correcoes, on="customerID", how="left")
    revisado["corrigido"] = (revisado["atrito_corrigido"].notna()
                             & (revisado["atrito_corrigido"] != revisado["atrito_laya"]))
    revisado["atrito"] = revisado["atrito_corrigido"].where(revisado["corrigido"], revisado["atrito_laya"])
    revisado = revisado.drop(columns="atrito_corrigido")
    revisado.to_csv(ARQUIVO_REVISADO, index=False)

    print(f"Lotes revisados: {len(arquivos)} | corrigidos: {revisado['corrigido'].sum()} de {len(revisado)} "
          f"({revisado['corrigido'].mean():.1%})")
    print(pd.crosstab(revisado["atrito_laya"], revisado["atrito"], margins=True))


if __name__ == "__main__":
    etapas = {"classificar": classificar, "lotes": lotes, "juntar": juntar}
    if len(sys.argv) != 2 or sys.argv[1] not in etapas:
        raise SystemExit("Uso: python laya_feedback.py [classificar | lotes | juntar]")
    etapas[sys.argv[1]]()
