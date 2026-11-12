# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "vaderSentiment",
#   "wordcloud",
# ]
# ///
# MAGIC %md
# MAGIC # Terra Signal: por que os clientes estão cancelando?
# MAGIC ### Análise exploratória: quem sai, quando e por quê
# MAGIC
# MAGIC **Situação.** A Terra Signal é uma operadora de telecom com cerca de 7 mil clientes de telefonia e internet.
# MAGIC
# MAGIC **Complicação.** No último período fiscal o churn chegou a **26,7%**. A empresa não sabe por que os clientes saem nem onde investir para reter.
# MAGIC
# MAGIC **Pergunta-chave.** Quais clientes estão saindo, por quê, e onde o churn pesa mais na receita?
# MAGIC
# MAGIC **Resposta em uma frase.** O churn tem **dois momentos**: nos primeiros 6 meses o cliente sai por **falha técnica** (sobretudo em DSL, mas também em fibra); depois disso, o cliente de **fibra em contrato mensal** sai por **preço e oferta da concorrência**. O **cheque eletrônico** é um fator de risco à parte, que vale mesmo controlando pelo contrato.
# MAGIC
# MAGIC | Bloco | Pergunta | O que mostra |
# MAGIC |---|---|---|
# MAGIC | 1 | Qual o tamanho do problema? | taxa, churners, receita mensal perdida |
# MAGIC | 2 | Qual o vínculo com a empresa? | tempo de casa e contrato |
# MAGIC | 3 | O que ele consome? | tipo de internet, serviços agregados, fibra e DSL recentes |
# MAGIC | 4 | Como ele paga? | cheque eletrônico, com controle por contrato |
# MAGIC | 5 | Quanto ele paga? | fatura mensal |
# MAGIC | 6 | Por que ele diz que saiu? | motivos do feedback, e como mudam com o tempo de casa |
# MAGIC | 7 | Síntese | quais hipóteses se confirmam |
# MAGIC | 8 | Onde o churn se concentra? | segmentos de internet × contrato × tempo de casa |
# MAGIC | 9 | O que os dados indicam | indícios para a modelagem e para a retenção |
# MAGIC | Apêndice | Detalhes e hipóteses descartadas | perfil, renda, streaming, sentimento etc. |
# MAGIC
# MAGIC **Regra sobre o feedback.** A pesquisa foi feita com clientes ativos e com quem já tinha cancelado, e não há data de resposta (informação da empresa). Por isso o texto é usado aqui só para **entender motivos**, nunca como variável preditiva.

# COMMAND ----------

# MAGIC %pip install vaderSentiment wordcloud

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Preparação
# MAGIC Carrega a base tratada, define o padrão visual e cria todas as variáveis derivadas em um só lugar (assim a síntese e o apêndice usam exatamente as mesmas definições).
# MAGIC
# MAGIC **Como ler cada análise:** para cada grupo, olhamos **taxa de churn**, **lift** (taxa do grupo ÷ taxa geral), **% do total de churners** e **receita mensal perdida** (soma de `MonthlyCharges` dos churners). O lift diz onde o risco é alto; volume e receita dizem onde vale agir.

# COMMAND ----------

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

TABELA_HIST = "workspace.default.history_tratado"
MOEDA = "US$"   # moeda da base; usada em todos os rótulos

def carregar(tabela, arquivo):
    try:
        return spark.table(tabela).toPandas()
    except NameError:  # fora do Databricks (teste local)
        return pd.read_csv(arquivo)

def tipar(d):
    for c in ["tenure", "TotalCharges", "MonthlyCharges", "MonthlyIncome", "SeniorCitizen",
              "fatura_sobre_renda", "tenure_missing", "tc_missing"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

df  = tipar(carregar(TABELA_HIST, "history_tratado.csv"))
df["churn"] = (df["Churn"] == "Yes").astype(int)

TAXA_GERAL = df["churn"].mean()
TOTAL_CHURNERS = df["churn"].sum()
RECEITA_PERDIDA_TOTAL = df.loc[df.churn == 1, "MonthlyCharges"].sum()

# COMMAND ----------

# ---------- Estilo visual ----------
DESTAQUE = "#2a78d6"   # grupos acima da média
CONTEXTO = "#c3c2b7"   # demais grupos
TEXTO    = "#0b0b0b"
TEXTO_2  = "#52514e"
FUNDO    = "#fcfcfb"
SERIES   = ["#2a78d6", "#eb6834", "#1baf7a"]  # até 3 séries, ordem fixa

plt.rcParams.update({
    "figure.facecolor": FUNDO, "axes.facecolor": FUNDO, "savefig.facecolor": FUNDO,
    "font.size": 11, "axes.edgecolor": CONTEXTO, "axes.labelcolor": TEXTO_2,
    "xtick.color": TEXTO_2, "ytick.color": TEXTO, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False,
})

def titulo(ax, titulo, subtitulo):
    ax.set_title(titulo, loc="left", fontsize=14, fontweight="bold", color=TEXTO, pad=28)
    ax.text(0, 1.02, subtitulo, transform=ax.transAxes, fontsize=10.5, color=TEXTO_2, va="bottom")

def dinheiro(v):
    return f"{MOEDA} {v:,.0f}"

# ---------- Padrão de leitura ----------
def resumo(data, col, ordem=None):
    t = (data.groupby(col, observed=True)
             .agg(clientes=("churn", "size"), churners=("churn", "sum"),
                  receita_perdida=("MonthlyCharges", lambda s: s[data.loc[s.index, "churn"] == 1].sum()))
             .reset_index())
    t["taxa"] = t.churners / t.clientes
    t["lift"] = t.taxa / TAXA_GERAL
    t["pct_churners"] = t.churners / TOTAL_CHURNERS
    if ordem is not None:
        t[col] = pd.Categorical(t[col], categories=ordem, ordered=True)
        t = t.sort_values(col)
    return t

def mostrar(t):
    fmt = {"taxa": "{:.1%}", "lift": "{:.2f}x", "pct_churners": "{:.1%}",
           "receita_perdida": MOEDA + " {:,.0f}", "clientes": "{:,}", "churners": "{:,}"}
    display(t.style.format({k: v for k, v in fmt.items() if k in t.columns}).hide(axis="index"))

def grafico_taxa(t, col, titulo_txt=None, subtitulo=None, ordenar=True, rotulo_col=None):
    """Barras horizontais: taxa de churn por grupo, destaque nos grupos acima da média."""
    t = t.sort_values("taxa") if ordenar else t.iloc[::-1]
    nomes = t[rotulo_col or col].astype(str)
    cores = [DESTAQUE if v > TAXA_GERAL * 1.1 else CONTEXTO for v in t.taxa]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(t) + 1.6))
    ax.barh(nomes, t.taxa, color=cores, height=0.6)
    ax.axvline(TAXA_GERAL, color=TEXTO_2, ls="--", lw=1)
    ax.text(TAXA_GERAL, len(t) - 0.4, f" média {TAXA_GERAL:.1%}", color=TEXTO_2, fontsize=9, va="bottom")
    for i, (taxa, n) in enumerate(zip(t.taxa, t.churners)):
        ax.text(taxa + 0.005, i, f"{taxa:.0%}  ({n:,} churners)", va="center", fontsize=10, color=TEXTO)
    ax.set_xlim(0, max(t.taxa.max() * 1.35, TAXA_GERAL * 1.5))
    ax.xaxis.set_visible(False); ax.spines["bottom"].set_visible(False); ax.tick_params(left=False)
    if titulo_txt is None:
        top = t.sort_values("taxa").iloc[-1]
        nome = str(top[rotulo_col or col])
        nome = f"{col} = {nome}" if nome in ("Yes", "No", "0", "1") else nome
        titulo_txt = f"{nome}: churn de {top.taxa:.0%}, {top.lift:.1f}x a média"
    titulo(ax, titulo_txt, subtitulo or f"Taxa de churn por {col}")
    plt.tight_layout(); plt.show()

# COMMAND ----------

# ---------- Variáveis derivadas (todas aqui, usadas no corpo, na síntese e no apêndice) ----------
AGREGADOS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
PRODUTOS  = ["PhoneService", "MultipleLines"] + AGREGADOS + ["StreamingTV", "StreamingMovies"]
ORDEM_TENURE = ["0–6 meses", "7–12", "13–24", "25–48", "49+"]
ORDEM_FAT    = [f"até {MOEDA} 35", "35–60", "60–80", "80–100", "100+"]
ORDEM_PAG    = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]

def derivar(d):
    d = d.copy()
    d["faixa_tenure"]  = pd.cut(d.tenure, [-1, 6, 12, 24, 48, 100], labels=ORDEM_TENURE)
    d["momento"]       = np.where(d.tenure <= 6, "Recente (0–6 meses)", "Mais de 6 meses")
    d["faixa_idade"]   = d.SeniorCitizen.map({1: "Idoso (65+)", 0: "Não idoso"})
    d["vinculo_familiar"] = np.select(
        [(d.Partner == "Yes") & (d.Dependents == "Yes"), d.Partner == "Yes", d.Dependents == "Yes"],
        ["Parceiro e dependentes", "Só parceiro", "Só dependentes"], default="Sem parceiro e sem dependentes")
    d["quintil_renda"] = pd.qcut(d.MonthlyIncome, 5, labels=["Q1 (menor)", "Q2", "Q3", "Q4", "Q5 (maior)"])
    d["dado_faltante"] = np.where((d.get("tenure_missing", 0) == 1) | (d.get("tc_missing", 0) == 1),
                                  "Com dado faltante", "Completo")
    d["qtd_agregados"] = np.where(d.InternetService == "No", np.nan,
                                  (d[AGREGADOS] == "Yes").sum(axis=1)).astype("float")
    d["streaming"] = np.select([(d.StreamingTV == "Yes") & (d.StreamingMovies == "Yes"),
                                (d.StreamingTV == "Yes") | (d.StreamingMovies == "Yes")],
                               ["TV e filmes", "Só um"], default="Nenhum")
    d["qtd_produtos"]   = (d[PRODUTOS] == "Yes").sum(axis=1) + (d.InternetService != "No").astype(int)
    d["faixa_produtos"] = pd.cut(d.qtd_produtos, [0, 1, 3, 5, 9], labels=["1", "2–3", "4–5", "6+"])
    d["faixa_fatura"]   = pd.cut(d.MonthlyCharges, [0, 35, 60, 80, 100, 200], labels=ORDEM_FAT)
    d["quintil_peso"]   = pd.qcut(d.MonthlyCharges / d.MonthlyIncome, 5,
                                  labels=["Q1 (pesa menos)", "Q2", "Q3", "Q4", "Q5 (pesa mais)"])
    d["cheque_eletronico"] = np.where(d.PaymentMethod == "Electronic check", "Cheque eletrônico", "Outros meios")
    return d

df  = derivar(df)
net = df[df.InternetService != "No"].copy()   # só clientes com internet
net["qtd_agregados"] = net.qtd_agregados.astype(int).astype(str)

print(f"Clientes: {len(df):,} | Churners: {TOTAL_CHURNERS:,} | Taxa geral: {TAXA_GERAL:.1%} | "
      f"Receita mensal perdida: {dinheiro(RECEITA_PERDIDA_TOTAL)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Qual o tamanho do problema?
# MAGIC Um em cada quatro clientes saiu. A receita mensal perdida é a soma das faturas de quem cancelou: é o que a empresa deixa de faturar **todo mês**.

# COMMAND ----------

fig, ax = plt.subplots(figsize=(9, 2.6))
ficou = len(df) - TOTAL_CHURNERS
ax.barh([""], [TOTAL_CHURNERS], color=DESTAQUE, height=0.5)
ax.barh([""], [ficou], left=[TOTAL_CHURNERS], color=CONTEXTO, height=0.5)
ax.text(TOTAL_CHURNERS / 2, 0, f"Saíram\n{TOTAL_CHURNERS:,} ({TAXA_GERAL:.1%})", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
ax.text(TOTAL_CHURNERS + ficou / 2, 0, f"Ficaram\n{ficou:,}", ha="center", va="center", color=TEXTO, fontsize=11)
ax.axis("off")
titulo(ax, f"{TAXA_GERAL:.0%} dos clientes cancelaram: {dinheiro(RECEITA_PERDIDA_TOTAL)} de receita perdida por mês",
       f"Base histórica, {len(df):,} clientes")
plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Qual o vínculo com a empresa?
# MAGIC **Hipóteses:** B1 o risco é maior nos primeiros meses; B2 contrato mensal sai mais; B3 o efeito do contrato não é só tempo de casa.

# COMMAND ----------

# MAGIC %md
# MAGIC ### B1. O risco é maior nos primeiros meses?

# COMMAND ----------

t = resumo(df, "faixa_tenure", ordem=ORDEM_TENURE); mostrar(t)

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(t.faixa_tenure.astype(str), t.taxa, color=DESTAQUE, lw=2.5, marker="o", ms=8)
ax.axhline(TAXA_GERAL, color=TEXTO_2, ls="--", lw=1)
ax.text(len(t) - 1, TAXA_GERAL, f"média {TAXA_GERAL:.1%}", color=TEXTO_2, fontsize=9, va="bottom", ha="right")
for x, y in zip(t.faixa_tenure.astype(str), t.taxa):
    ax.annotate(f"{y:.0%}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10)
ax.yaxis.set_visible(False); ax.spines["left"].set_visible(False); ax.set_ylim(0, t.taxa.max() * 1.25)
p = t.iloc[0]
titulo(ax, f"Até 6 meses de casa, churn de {p.taxa:.0%}: {p.pct_churners:.0%} de todos os churners",
       f"Taxa de churn por tempo de casa (tenure) · {p.lift:.1f}x a média")
plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### B2. Contrato mensal sai mais?

# COMMAND ----------

t = resumo(df, "Contract"); mostrar(t)
mm = t.set_index("Contract").loc["Month-to-month"]
grafico_taxa(t, "Contract",
             titulo_txt=f"Contrato mensal: churn de {mm.taxa:.0%} e {mm.pct_churners:.0%} de todos os churners")

# COMMAND ----------

# MAGIC %md
# MAGIC ### B3. O efeito do contrato é só tempo de casa?
# MAGIC Se as linhas não se encontram em nenhuma faixa de tenure, o contrato tem efeito próprio. Contexto da empresa: **não há multa** nos contratos anual e bianual. Então a diferença entre os contratos não vem de barreira de saída.

# COMMAND ----------

ct = df.groupby(["Contract", "faixa_tenure"], observed=True).churn.agg(["mean", "size"]).reset_index()
fig, ax = plt.subplots(figsize=(9, 4.5))
for cor, contrato in zip(SERIES, ["Month-to-month", "One year", "Two year"]):
    s = ct[(ct.Contract == contrato) & (ct["size"] >= 30)]
    ax.plot(s.faixa_tenure.astype(str), s["mean"], color=cor, lw=2.5, marker="o", ms=8)
    ax.text(len(ORDEM_TENURE) - 1 + 0.1, s["mean"].iloc[-1], contrato, color=TEXTO, va="center", fontsize=10)
ax.set_xlim(-0.3, len(ORDEM_TENURE) + 0.8)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
mm_min = ct[(ct.Contract == "Month-to-month") & (ct["size"] >= 30)]["mean"].min()
titulo(ax, f"Mesmo com anos de casa, contrato mensal tem churn de pelo menos {mm_min:.0%}",
       "Taxa de churn por tempo de casa, uma linha por tipo de contrato (grupos com 30+ clientes)")
plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. O que o cliente consome?
# MAGIC **Hipóteses:** C1 fibra sai mais que DSL; C2 quem não tem proteção e suporte sai mais; C5 o começo da relação é crítico nos dois tipos de internet.

# COMMAND ----------

# MAGIC %md
# MAGIC ### C1. Clientes de fibra saem mais?

# COMMAND ----------

t = resumo(df, "InternetService"); mostrar(t); grafico_taxa(t, "InternetService")

# COMMAND ----------

# MAGIC %md
# MAGIC ### C2. Quem não tem proteção e suporte sai mais?
# MAGIC Só entre clientes com internet. Primeiro cada serviço, depois a quantidade.

# COMMAND ----------

linhas = []
for s in AGREGADOS:
    for v in ["No", "Yes"]:
        g = net[net[s] == v]
        linhas.append({"grupo": f"{s} = {'não tem' if v == 'No' else 'tem'}", "taxa": g.churn.mean(),
                       "churners": g.churn.sum(), "clientes": len(g), "lift": g.churn.mean() / TAXA_GERAL})
grafico_taxa(pd.DataFrame(linhas), "grupo", subtitulo="Taxa de churn por serviço agregado, só clientes com internet")

t = resumo(net, "qtd_agregados", ordem=["0", "1", "2", "3", "4"]); mostrar(t)
grafico_taxa(t, "qtd_agregados", ordenar=False,
             titulo_txt=f"Sem nenhum serviço agregado o churn é {t.taxa.iloc[0]:.0%}; com os 4, {t.taxa.iloc[-1]:.0%}",
             subtitulo="Taxa de churn por quantidade de serviços agregados (segurança, backup, proteção, suporte)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### C5. O começo da relação é crítico nos dois tipos de internet?
# MAGIC A análise de feedback apontou falha técnica em DSL recente. Aqui olhamos a **taxa**: o pico dos primeiros meses aparece em DSL e também em fibra, e fibra recente é o maior bolsão de churners.

# COMMAND ----------

ti = (df[df.InternetService != "No"]
        .groupby(["InternetService", "faixa_tenure"], observed=True)
        .agg(clientes=("churn", "size"), churners=("churn", "sum"), taxa=("churn", "mean")).reset_index())
display(ti.pivot(index="faixa_tenure", columns="InternetService", values=["taxa", "churners"]))

fig, ax = plt.subplots(figsize=(9, 4.5))
for cor, (serv, nome) in zip(SERIES, [("Fiber optic", "Fibra"), ("DSL", "DSL")]):
    s = ti[ti.InternetService == serv]
    ax.plot(s.faixa_tenure.astype(str), s.taxa, color=cor, lw=2.5, marker="o", ms=8)
    ax.text(len(ORDEM_TENURE) - 1 + 0.1, s.taxa.iloc[-1], nome, color=TEXTO, va="center", fontsize=10)
    ax.annotate(f"{s.taxa.iloc[0]:.0%}", (0, s.taxa.iloc[0]), textcoords="offset points", xytext=(-22, 0), va="center", fontsize=10)
ax.axhline(TAXA_GERAL, color=TEXTO_2, ls="--", lw=1)
ax.set_xlim(-0.6, len(ORDEM_TENURE) + 0.5)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
fr = ti[(ti.InternetService == "Fiber optic") & (ti.faixa_tenure == ORDEM_TENURE[0])].iloc[0]
dr = ti[(ti.InternetService == "DSL") & (ti.faixa_tenure == ORDEM_TENURE[0])].iloc[0]
titulo(ax, f"Nos primeiros 6 meses saem {fr.taxa:.0%} dos clientes de fibra e {dr.taxa:.0%} dos de DSL",
       f"Taxa de churn por tempo de casa e tipo de internet · fibra recente: {int(fr.churners)} churners, DSL recente: {int(dr.churners)}")
plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Como o cliente paga?
# MAGIC **Hipótese E1:** cheque eletrônico sai mais. Atenção: o risco é do **cheque eletrônico**, não de "pagamento manual" em geral. Cheque por correio também é manual e fica abaixo da média.

# COMMAND ----------

t = resumo(df, "PaymentMethod"); mostrar(t)
ec = t.set_index("PaymentMethod").loc["Electronic check"]
grafico_taxa(t, "PaymentMethod",
             titulo_txt=f"Cheque eletrônico: churn de {ec.taxa:.0%} e {ec.pct_churners:.0%} dos churners",
             subtitulo="Taxa de churn por meio de pagamento · cheque por correio, também manual, fica abaixo da média")

# COMMAND ----------

# MAGIC %md
# MAGIC #### O efeito do cheque eletrônico se mantém dentro do mesmo contrato?
# MAGIC Quem paga com cheque eletrônico é mais mensal. Se o efeito sumisse dentro do mesmo contrato, seria só reflexo do contrato. Não some.

# COMMAND ----------

pc = (df.groupby(["Contract", "cheque_eletronico"]).churn.agg(["mean", "size"]).reset_index())
display(pc.pivot(index="Contract", columns="cheque_eletronico", values=["mean", "size"]))

piv = pc.pivot(index="Contract", columns="cheque_eletronico", values="mean").reindex(["Month-to-month", "One year", "Two year"])
fig, ax = plt.subplots(figsize=(9, 3.8))
y = np.arange(len(piv))[::-1]
ax.hlines(y, piv["Outros meios"], piv["Cheque eletrônico"], color=CONTEXTO, lw=3)
ax.scatter(piv["Outros meios"], y, color=CONTEXTO, s=90, zorder=3)
ax.scatter(piv["Cheque eletrônico"], y, color=DESTAQUE, s=90, zorder=3)
for yi, (o, e) in zip(y, zip(piv["Outros meios"], piv["Cheque eletrônico"])):
    ax.text(e, yi + 0.22, f"{e:.0%}", color=DESTAQUE, ha="center", fontsize=9.5)
    ax.text(o, yi + 0.22, f"{o:.0%}", color=TEXTO_2, ha="center", fontsize=9.5)
ax.set_yticks(y, piv.index); ax.tick_params(left=False); ax.set_ylim(-0.6, len(piv) - 0.3)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
m = piv.loc["Month-to-month"]
titulo(ax, f"No mensal, cheque eletrônico cancela {m['Cheque eletrônico']:.0%} vs {m['Outros meios']:.0%} dos outros meios",
       "Taxa de churn por contrato: azul = cheque eletrônico, cinza = demais meios de pagamento")
plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Quanto o cliente paga?
# MAGIC **Hipótese D1:** fatura alta leva ao churn. O churn sobe com a fatura, mas **não é linear**: recua na faixa mais cara (que concentra clientes antigos com muitos produtos). Preço sozinho não explica; o que o feedback mostra é **percepção de valor** frente à concorrência (bloco 6). O apêndice D mostra que, com o mesmo tipo de internet e a mesma quantidade de produtos, quem sai paga praticamente o mesmo que quem fica.

# COMMAND ----------

t = resumo(df, "faixa_fatura", ordem=ORDEM_FAT); mostrar(t)
pico = t.sort_values("taxa").iloc[-1]
grafico_taxa(t, "faixa_fatura", ordenar=False,
             titulo_txt=f"O churn sobe até a faixa {pico.faixa_fatura} ({pico.taxa:.0%}) e recua na mais cara",
             subtitulo=f"Taxa de churn por faixa de fatura mensal ({MOEDA})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Por que o cliente diz que saiu?
# MAGIC Categorias de motivo identificadas por palavras-chave (regex) no texto, com as regras revisadas por auditoria amostral (25 feedbacks por categoria; regras de pagamento e de produto/cobertura corrigidas depois da revisão). Um feedback pode ter mais de um motivo.
# MAGIC
# MAGIC Uso **descritivo**: o feedback explica motivos, não entra no modelo.

# COMMAND ----------

texto = (df.CustomerFeedback.fillna("")
           .str.replace(r"[\r\n]+", " ", regex=True).str.strip().str.lower())

def tem(rx):
    return texto.str.contains(rx, regex=True)

REGRA_PRECO = (r"better deal|better value|more affordable|cost savings|too expensive|high monthly charges|charges.*too high"
               r"|competitive pric|cost considerations|cheaper|pricing.*(?:led|reason|ultimately)")
REGRA_TECNICO = (r"slow speed|unreliable|connectivity issue|connection issue|connection problem|service did not meet"
                 r"|speed.*did not meet|outage")
PAG_SEM_PROBLEMA = r"(?:never had|did not encounter|no).{0,30}(?:issues?|problems?) with billing"
PAG_PROBLEMA = [
    # meio de pagamento inconveniente, limitado ou ultrapassado
    r"(?:payment method|payment options?|billing options?|electronic check|mailed check|bank transfer|credit card).{0,60}"
    r"(?:inconvenient|outdated|limited|lack of flexibility|not flexible|more flexible|more convenient|hassle(?!-free))",
    r"(?:inconvenient|outdated|limited|lack of flexibility|not flexible|hassle(?!-free)).{0,60}"
    r"(?:payment method|payment options?|electronic check|mailed check|bank transfer|credit card)",
    # falhas explícitas de processamento
    r"(?:payment|electronic check|mailed check|bank transfer|credit card).{0,60}"
    r"(?:did not go through|didn.t go through|not processed|not always processed|took longer than expected|not processed in a timely manner)",
    # problemas associados ao meio de pagamento
    r"(?:issues?|problems?|delays?).{0,30}(?:with|using|in).{0,30}(?:payment method|electronic check|mailed check|bank transfer|credit card)",
    r"billing issues|billing problems|billing delays|issues with billing|problems with billing|delays with billing",
    # obrigação de usar cheque pelo correio
    r"(?:had to|having to|limited to).{0,40}(?:mailing a check|mailed check)",
    # preferência explícita por outro meio
    r"prefer.{0,60}(?:credit card|online payment|automatic payment).{0,80}(?:rather than|instead of|over).{0,50}(?:electronic check|mailed check)",
]
REGRA_PRODUTO = (r"lack of internet service|lack of internet access|without internet service|without internet access"
                 r"|no access to internet|did not have internet service|internet service was not included|internet service not included"
                 r"|internet service was not available|internet service not available|internet service was not provided"
                 r"|internet service not provided|did not offer internet service|not offer internet service"
                 r"|lack of internet service options|not available in my area|internet service options available"
                 r"|no internet service.*(?:inconvenience|inconvenient|not satisfied|disappointed)")
REGRA_PESSOAL = r"personal reasons|financial issues|moving|move to|relocat"
REGRA_ATENDIMENTO = r"customer service"

MOTIVOS = {
    "Preço / concorrência":     tem(REGRA_PRECO),
    "Qualidade técnica":        tem(REGRA_TECNICO),
    "Experiência de pagamento": ~tem(PAG_SEM_PROBLEMA) & tem("|".join(f"(?:{r})" for r in PAG_PROBLEMA)),
    "Produto / cobertura":      tem(REGRA_PRODUTO),
    "Pessoal / financeiro":     tem(REGRA_PESSOAL),
    "Atendimento":              tem(REGRA_ATENDIMENTO),
}
COL_MOTIVO = {m: "mot_" + re.sub(r"\W+", "_", m.lower()).strip("_") for m in MOTIVOS}
for m, flag in MOTIVOS.items():
    df[COL_MOTIVO[m]] = flag.astype(int)
df["mot_algum"] = df[list(COL_MOTIVO.values())].max(axis=1)

ch = df[df.churn == 1]
mot = pd.Series({m: ch[c].mean() for m, c in COL_MOTIVO.items()}).sort_values()
print(f"Churners sem nenhum motivo identificado: {1 - ch.mot_algum.mean():.0%}")

fig, ax = plt.subplots(figsize=(9, 0.55 * len(mot) + 1.6))
ax.barh(mot.index, mot.values, color=[DESTAQUE if v == mot.max() else CONTEXTO for v in mot.values], height=0.6)
for i, v in enumerate(mot.values):
    ax.text(v + 0.005, i, f"{v:.0%}", va="center", fontsize=10)
ax.set_xlim(0, mot.max() * 1.25); ax.xaxis.set_visible(False); ax.spines["bottom"].set_visible(False); ax.tick_params(left=False)
titulo(ax, f"{mot.idxmax()} é o motivo mais citado: {mot.max():.0%} dos churners",
       "% de churners que citam cada motivo (um feedback pode citar mais de um)")
plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### O motivo muda com o tempo de casa
# MAGIC O motivo de saída depende do momento da relação: no começo, o cliente sai por falha técnica; depois, por preço e concorrência.

# COMMAND ----------

C_PRECO, C_TEC = COL_MOTIVO["Preço / concorrência"], COL_MOTIVO["Qualidade técnica"]
mt = (ch[ch.InternetService != "No"]
        .groupby(["InternetService", "faixa_tenure"], observed=True)
        .agg(churners=("churn", "size"), preco=(C_PRECO, "mean"), tecnico=(C_TEC, "mean")).reset_index())
display(mt)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for ax, (serv, nome) in zip(axes, [("DSL", "DSL"), ("Fiber optic", "Fibra")]):
    s = mt[mt.InternetService == serv]
    xs = s.faixa_tenure.astype(str)
    for cor, col, rot in [(SERIES[0], "tecnico", "Qualidade técnica"), (SERIES[1], "preco", "Preço / concorrência")]:
        ax.plot(xs, s[col], color=cor, lw=2.5, marker="o", ms=7)
        ax.text(len(xs) - 1 + 0.1, s[col].iloc[-1], rot, color=cor, va="center", fontsize=9.5)
    ax.set_xlim(-0.3, len(xs) + 1.3)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_title(f"{nome} · {s.churners.sum():,} churners", loc="left", fontsize=11.5, fontweight="bold", color=TEXTO_2)
d0 = mt[(mt.InternetService == "DSL") & (mt.faixa_tenure == ORDEM_TENURE[0])].iloc[0]
f_rest = ch[(ch.InternetService == "Fiber optic") & (ch.tenure > 6)][C_PRECO].mean()
fig.suptitle(f"Até 6 meses, {d0.tecnico:.0%} dos churners de DSL citam falha técnica; "
             f"depois disso, {f_rest:.0%} dos churners de fibra citam preço ou concorrência",
             x=0.01, ha="left", fontsize=13.5, fontweight="bold", color=TEXTO)
fig.text(0.01, 0.89, "% de churners que citam cada motivo, por tempo de casa", fontsize=10.5, color=TEXTO_2)
plt.tight_layout(rect=(0, 0, 1, 0.88)); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Os primeiros meses em detalhe: 0–3 × 4–6 meses
# MAGIC Abrindo o começo da relação em faixas menores: até 3 meses predominam falha técnica e atrito de pagamento; entre 4 e 6 meses, preço e concorrência já pesam tanto quanto o técnico; a partir de 7 meses, preço vira o tema principal.

# COMMAND ----------

ORDEM_TENURE_FINA = ["0–3", "4–6", "7–12", "13–24", "25–36", "37+"]
df["faixa_tenure_fina"] = pd.cut(df.tenure, [-1, 3, 6, 12, 24, 36, 100], labels=ORDEM_TENURE_FINA)
C_PAG = COL_MOTIVO["Experiência de pagamento"]
tf = (df.groupby("faixa_tenure_fina", observed=True)
        .agg(clientes=("churn", "size"), churners=("churn", "sum"), taxa=("churn", "mean")))
tf = tf.join(df[df.churn == 1].groupby("faixa_tenure_fina", observed=True)[[C_TEC, C_PRECO, C_PAG]].mean()
               .rename(columns={C_TEC: "% técnico", C_PRECO: "% preço", C_PAG: "% pagamento"}))
display(tf.style.format({"taxa": "{:.0%}", "% técnico": "{:.0%}", "% preço": "{:.0%}", "% pagamento": "{:.0%}"}))

fig, ax = plt.subplots(figsize=(9, 4.5))
xs = tf.index.astype(str)
for cor, col in zip(SERIES, ["% técnico", "% preço", "% pagamento"]):
    ax.plot(xs, tf[col], color=cor, lw=2.5, marker="o", ms=7)
    ax.text(len(xs) - 1 + 0.1, tf[col].iloc[-1], col.replace("% ", "").capitalize(), color=cor, va="center", fontsize=10)
ax.set_xlim(-0.3, len(xs) + 0.9)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
ax.set_xlabel("Tempo de casa (meses)")
r03 = tf.loc["0–3"]
titulo(ax, f"Até 3 meses: {r03['% técnico']:.0%} dos churners citam falha técnica e {r03['% pagamento']:.0%}, pagamento",
       f"% de churners que citam cada motivo, por tempo de casa · churn de {r03.taxa:.0%} até 3 meses e {tf.loc['4–6'].taxa:.0%} entre 4 e 6")
plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preço: valor alto ou oferta melhor da concorrência?
# MAGIC Entre os churners que citam preço, separa quem diz que a fatura é alta de quem cita uma oferta melhor em outra operadora (pode ser os dois). Considera só os churners com menção explícita a uma das duas coisas. Também mede quanto de **toda a base** de cada segmento saiu citando preço.

# COMMAND ----------

df["preco_alto"] = tem(r"high monthly charges|charges.*too high|monthly charges.*too high|too expensive|bit on the higher side|pricing.*too high").astype(int)
df["oferta_concorrente"] = tem(r"better deal|better value|more affordable|cost savings|competitive pric|cheaper").astype(int)

pr = df[(df.churn == 1) & (df[C_PRECO] == 1) & ((df.preco_alto == 1) | (df.oferta_concorrente == 1))]  # só menções explícitas
seg_preco = (pr.groupby(["InternetService", "Contract"])
               .agg(churners_preco=("churn", "size"), cita_preco_alto=("preco_alto", "mean"),
                    cita_concorrente=("oferta_concorrente", "mean"), fatura_media=("MonthlyCharges", "mean"),
                    tenure_medio=("tenure", "mean")))
base_seg = df.groupby(["InternetService", "Contract"]).agg(clientes=("churn", "size"), churners=("churn", "sum"))
seg_preco = seg_preco.join(base_seg)
seg_preco["% da base que saiu por preço"] = seg_preco.churners_preco / seg_preco.clientes
seg_preco["% dos churners por preço"] = seg_preco.churners_preco / seg_preco.churners
seg_preco = seg_preco[seg_preco.churners_preco >= 10].sort_values("churners_preco", ascending=False)
display(seg_preco.style.format({"cita_preco_alto": "{:.0%}", "cita_concorrente": "{:.0%}", "fatura_media": MOEDA + " {:,.2f}",
                                "tenure_medio": "{:.0f}", "% da base que saiu por preço": "{:.1%}", "% dos churners por preço": "{:.0%}"}))

fm = seg_preco.loc[("Fiber optic", "Month-to-month")]
fig, ax = plt.subplots(figsize=(10, 2.8))
vals = pd.Series({"Cita oferta melhor da concorrência": fm.cita_concorrente, "Diz que o preço é alto": fm.cita_preco_alto})
ax.barh(vals.index[::-1], vals.values[::-1], color=[CONTEXTO, DESTAQUE], height=0.55)
for i, v in enumerate(vals.values[::-1]):
    ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=10)
ax.set_xlim(0, 1.1); ax.xaxis.set_visible(False); ax.spines["bottom"].set_visible(False); ax.tick_params(left=False)
titulo(ax, f"Fibra mensal: {fm['% da base que saiu por preço']:.1%} da base saiu citando preço; {fm.cita_concorrente:.0%} citam a concorrência",
       f"Churners de fibra mensal que citam preço ({int(fm.churners_preco)}), por tipo de menção · pode citar os dois")
plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pagamento aparece junto com outros motivos
# MAGIC Quem reclama do pagamento quase sempre cita outro motivo também: o pagamento raramente é a única causa da saída.
# MAGIC
# MAGIC **Reclamar não é o mesmo que ter risco.** As reclamações de pagamento vêm quase todas de meios manuais, incluindo o cheque por correio. Mas quem paga com cheque por correio cancela abaixo da média. O meio com risco de saída é o cheque eletrônico.

# COMMAND ----------

C_PAG = COL_MOTIVO["Experiência de pagamento"]
outros = [c for c in COL_MOTIVO.values() if c != C_PAG]
pag = ch[ch[C_PAG] == 1]
junto = (pag[outros].max(axis=1) == 1).mean()
print(f"Churners que reclamam do pagamento: {len(pag):,} ({len(pag) / len(ch):.0%} dos churners)")
print(f"Desses, citam também outro motivo: {junto:.0%}")

reclama_vs_risco = pd.DataFrame({
    "% das reclamações de pagamento": pag.PaymentMethod.value_counts(normalize=True),
    "taxa de churn do meio": df.groupby("PaymentMethod").churn.mean(),
}).reindex(ORDEM_PAG)
display(reclama_vs_risco.style.format("{:.0%}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Síntese: quais hipóteses se confirmam?
# MAGIC **Critério:**
# MAGIC - **Não se confirma:** diferença entre o grupo de maior e o de menor taxa abaixo de 5 p.p.
# MAGIC - **Confirmada:** grupo de maior risco com lift ≥ 1,3, pelo menos 100 churners, e, para variáveis ordenadas, relação que só sobe ou só desce, e sem confusão relevante com outro fator.
# MAGIC - **Parcial:** os demais casos. A coluna `observação` diz por quê.
# MAGIC
# MAGIC O sentimento do feedback **não entra** na síntese: é consequência da saída para parte dos clientes (pesquisa feita também com quem já tinha cancelado), não causa.

# COMMAND ----------

def monotonica(taxas, tol=0.01):
    d = np.diff(np.asarray(taxas, dtype=float))
    return bool((d >= -tol).all() or (d <= tol).all())

# (código, hipótese, base, coluna, ordem se ordenada, observação manual que força "Parcial", onde está)
HIPOTESES = [
    ("B1", "Risco maior nos primeiros meses",   df,  "faixa_tenure",   ORDEM_TENURE, None, "Bloco 2"),
    ("B2", "Contrato mensal sai mais",          df,  "Contract",       ["Month-to-month", "One year", "Two year"], None, "Bloco 2"),
    ("C1", "Fibra sai mais",                    df,  "InternetService", None, None, "Bloco 3"),
    ("C2", "Sem proteção/suporte sai mais",     net, "qtd_agregados",  ["0", "1", "2", "3", "4"], None, "Bloco 3"),
    ("E1", "Cheque eletrônico sai mais",        df,  "PaymentMethod",  None, None, "Bloco 4"),
    ("D1", "Fatura alta leva ao churn",         df,  "faixa_fatura",   ORDEM_FAT, None, "Bloco 5"),
    ("A1", "Idosos saem mais",                  df,  "faixa_idade",    None,
           "confundido: idosos têm mais fibra, mensal e cheque eletrônico", "Apêndice A"),
    ("A2", "Sem parceiro/dependentes sai mais", df,  "vinculo_familiar", None, None, "Apêndice A"),
    ("A3", "Menor renda sai mais",              df,  "quintil_renda",  ["Q1 (menor)", "Q2", "Q3", "Q4", "Q5 (maior)"],
           "renda de base externa, sem garantia de precisão", "Apêndice A"),
    ("A4", "Gênero (controle)",                 df,  "gender",         None, None, "Apêndice A"),
    ("B4", "Dado faltante tem perfil diferente", df, "dado_faltante",  None, None, "Apêndice B"),
    ("C3", "Streaming reduz churn",             net, "streaming",      None, None, "Apêndice C"),
    ("C4", "Mais produtos retêm mais",          df,  "faixa_produtos", ["1", "2–3", "4–5", "6+"], None, "Apêndice C"),
    ("D3", "Fatura pesa no orçamento",          df,  "quintil_peso",   ["Q1 (pesa menos)", "Q2", "Q3", "Q4", "Q5 (pesa mais)"],
           "efeito some dentro da mesma faixa de fatura", "Apêndice D"),
    ("E2", "Fatura digital sai mais",           df,  "PaperlessBilling", None, None, "Apêndice E"),
]

linhas = []
for cod, hip, base, col, ordem, nota, onde in HIPOTESES:
    t = resumo(base, col, ordem=ordem)
    t50 = t[t.clientes >= 50]
    big, small = t50.sort_values("lift").iloc[-1], t50.sort_values("lift").iloc[0]
    amplitude = big.taxa - small.taxa
    obs = []
    mono = True
    if ordem is not None:
        mono = monotonica(t.taxa)
        if not mono:
            obs.append("relação não é contínua (sobe e desce)")
    if nota:
        obs.append(nota)
    if not obs and amplitude >= 0.05 and (big.lift < 1.3 or big.churners < 100):
        obs.append("lift abaixo de 1,3" if big.lift < 1.3 else "menos de 100 churners")
    if amplitude < 0.05:
        veredito = "Não se confirma"
    elif big.lift >= 1.3 and big.churners >= 100 and mono and not nota:
        veredito = "Confirmada"
    else:
        veredito = "Parcial"
    linhas.append({"código": cod, "hipótese": f"{cod}. {hip}", "grupo de maior risco": f"{col} = {big[col]}",
                   "taxa": big.taxa, "lift": big.lift, "churners": int(big.churners),
                   "receita_perdida": big.receita_perdida, "amplitude": amplitude,
                   "veredito": veredito, "observação": "; ".join(obs), "onde": onde})

res = pd.DataFrame(linhas).sort_values(["veredito", "lift"], ascending=[True, False])
display(res.drop(columns="código").style.format({"taxa": "{:.0%}", "lift": "{:.2f}x", "amplitude": "{:.0%}",
                                                  "receita_perdida": MOEDA + " {:,.0f}"}).hide(axis="index"))

fig, ax = plt.subplots(figsize=(10, 0.42 * len(res) + 1.6))
r = res.sort_values("lift")
cores = [DESTAQUE if v == "Confirmada" else CONTEXTO for v in r.veredito]
ax.barh(r["hipótese"], r.lift, color=cores, height=0.6)
ax.axvline(1, color=TEXTO_2, ls="--", lw=1); ax.text(1, len(r) - 0.4, " sem efeito (1x)", color=TEXTO_2, fontsize=9)
for i, (l, g, v) in enumerate(zip(r.lift, r["grupo de maior risco"], r.veredito)):
    ax.text(l + 0.03, i, f"{l:.1f}x  {g}" + ("" if v == "Confirmada" else f"  ({v.lower()})"), va="center", fontsize=9)
ax.set_xlim(0, r.lift.max() * 2.1); ax.xaxis.set_visible(False); ax.spines["bottom"].set_visible(False); ax.tick_params(left=False)
conf = res[res.veredito == "Confirmada"].sort_values("lift", ascending=False)
titulo(ax, f"{len(conf)} de {len(res)} hipóteses se confirmam",
       "Lift do grupo de maior risco (azul = confirmada): tempo de casa, contrato, fibra, suporte e cheque eletrônico")
plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Onde o churn se concentra?
# MAGIC Cruza os três fatores com mais sinal (tipo de internet, contrato e tempo de casa) em segmentos que não se sobrepõem. Para cada segmento: risco, volume, receita perdida, o motivo que os churners citam e quanto do segmento paga com cheque eletrônico. Segmentos com menos de 50 clientes ficam de fora.

# COMMAND ----------

NOME_NET = {"Fiber optic": "Fibra", "DSL": "DSL", "No": "Sem internet"}
df["segmento"] = (df.InternetService.map(NOME_NET) + " · "
                  + np.where(df.Contract == "Month-to-month", "mensal", "anual/bianual") + " · "
                  + np.where(df.tenure <= 6, "até 6 meses", "mais de 6 meses"))

seg = resumo(df, "segmento")
mot_s = df[df.churn == 1].groupby("segmento")[[C_TEC, C_PRECO, C_PAG]].mean()
mot_s.columns = ["% cita técnico", "% cita preço", "% cita pagamento"]
seg = seg.merge(mot_s, left_on="segmento", right_index=True, how="left")
seg["% cheque eletrônico"] = seg.segmento.map(df.groupby("segmento").PaymentMethod.apply(lambda s: (s == "Electronic check").mean()))
seg = seg[seg.clientes >= 50].sort_values("receita_perdida", ascending=False)
display(seg.style.format({"taxa": "{:.0%}", "lift": "{:.2f}x", "pct_churners": "{:.0%}",
                          "receita_perdida": MOEDA + " {:,.0f}", "% cita técnico": "{:.0%}", "% cita preço": "{:.0%}",
                          "% cita pagamento": "{:.0%}", "% cheque eletrônico": "{:.0%}"}).hide(axis="index"))

top = seg.head(8).sort_values("receita_perdida")
fig, ax = plt.subplots(figsize=(10, 0.6 * len(top) + 1.6))
cores = [DESTAQUE if v > TAXA_GERAL * 1.1 else CONTEXTO for v in top.taxa]
ax.barh(top.segmento, top.receita_perdida, color=cores, height=0.6)
for i, (rp, tx, n, tec, pr) in enumerate(zip(top.receita_perdida, top.taxa, top.churners, top["% cita técnico"], top["% cita preço"])):
    motivo = ("técnico" if tec >= pr else "preço") if max(tec, pr) >= 0.3 else "sem motivo dominante"
    ax.text(rp * 1.01 + 200, i, f"{dinheiro(rp)}/mês · churn {tx:.0%} · {n} churners · motivo: {motivo}",
            va="center", fontsize=9.5)
ax.set_xlim(0, top.receita_perdida.max() * 2.7); ax.xaxis.set_visible(False)
ax.spines["bottom"].set_visible(False); ax.tick_params(left=False)
top2 = seg.head(2)
titulo(ax, f"{top2.segmento.iloc[0].rsplit(' · ', 1)[0]} e {top2.segmento.iloc[1].rsplit(' · ', 1)[0]}: {top2.receita_perdida.sum() / RECEITA_PERDIDA_TOTAL:.0%} da receita perdida"
           if top2.segmento.iloc[0].rsplit(" · ", 1)[0] != top2.segmento.iloc[1].rsplit(" · ", 1)[0]
           else f"{top2.segmento.iloc[0].rsplit(' · ', 1)[0].replace(' · ', ' no contrato ')}: {top2.receita_perdida.sum() / RECEITA_PERDIDA_TOTAL:.0%} da receita perdida",
       f"Receita mensal perdida por segmento, top 8 (azul = churn acima da média) · {top2.pct_churners.sum():.0%} dos churners")
plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. O que os dados indicam
# MAGIC **Indícios para a modelagem**
# MAGIC - Variáveis com sinal forte: tempo de casa, contrato, tipo de internet, serviços agregados (sobretudo suporte e segurança) e cheque eletrônico.
# MAGIC - Variáveis fracas: gênero, renda (fonte externa, sem efeito), peso da fatura na renda (repete a fatura) e streaming.
# MAGIC - Feedback e sentimento não devem virar variáveis: parte das respostas veio de quem já tinha cancelado.
# MAGIC
# MAGIC **Indícios para a retenção**
# MAGIC - **Quando:** o risco se concentra nos primeiros 6 meses (52% de churn, 41% dos churners) e cai com o tempo de casa.
# MAGIC - **Onde:** fibra em contrato mensal concentra a maior parte da receita perdida, tanto no início da relação quanto depois.
# MAGIC - **Por quê:** no início, quem sai cita falha técnica (sobretudo em DSL); depois de 6 meses, quem sai da fibra cita preço e concorrência.
# MAGIC - **Contrato:** o mensal tem churn alto em qualquer tempo de casa, e o anual não tem multa. A diferença não vem de barreira de saída.
# MAGIC - **Pagamento:** o cheque eletrônico tem risco próprio, mas quem reclama do pagamento quase sempre cita outro motivo junto.
# MAGIC - **Suporte e segurança:** quem não tem nenhum serviço agregado sai 56%; com os quatro, 6%.
# MAGIC
# MAGIC Valores em `US$`, a moeda da base (variável `MOEDA` no topo).

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Apêndice
# MAGIC Detalhes que sustentam a síntese e hipóteses que não se confirmaram.

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Quem é o cliente?
# MAGIC ### A1. Idosos saem mais? (sim, mas boa parte é perfil)

# COMMAND ----------

t = resumo(df, "faixa_idade"); mostrar(t)
perfil_idade = df.groupby("faixa_idade").apply(lambda x: pd.Series({
    "% fibra": (x.InternetService == "Fiber optic").mean(),
    "% mensal": (x.Contract == "Month-to-month").mean(),
    "% cheque eletrônico": (x.PaymentMethod == "Electronic check").mean()}))
display(perfil_idade.style.format("{:.0%}"))

mf = df[(df.Contract == "Month-to-month") & (df.InternetService == "Fiber optic")]
t_mf = resumo(mf, "faixa_idade"); mostrar(t_mf)
a = t_mf.set_index("faixa_idade").taxa
grafico_taxa(t_mf, "faixa_idade",
             titulo_txt=f"Em fibra mensal, idosos saem {a['Idoso (65+)']:.0%} vs {a['Não idoso']:.0%}: a diferença encolhe",
             subtitulo="Taxa de churn por faixa de idade, só fibra + contrato mensal")

# COMMAND ----------

# MAGIC %md
# MAGIC ### A2. Quem não tem parceiro nem dependentes sai mais?

# COMMAND ----------

t = resumo(df, "vinculo_familiar"); mostrar(t); grafico_taxa(t, "vinculo_familiar")

# COMMAND ----------

# MAGIC %md
# MAGIC ### A3. Clientes de menor renda saem mais?
# MAGIC `MonthlyIncome` veio de uma base externa, sem garantia de precisão (informação da empresa). Não há efeito: a correlação com o churn é praticamente zero.

# COMMAND ----------

t = resumo(df, "quintil_renda", ordem=["Q1 (menor)", "Q2", "Q3", "Q4", "Q5 (maior)"]); mostrar(t)
print(f"Correlação renda x churn: {df[['MonthlyIncome', 'churn']].corr().iloc[0, 1]:.3f}")
grafico_taxa(t, "quintil_renda", ordenar=False,
             titulo_txt=f"Renda não muda o churn: de {t.taxa.min():.0%} a {t.taxa.max():.0%} entre quintis",
             subtitulo="Taxa de churn por quintil de renda mensal (fonte externa)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### A4. Gênero explica o churn? (controle)

# COMMAND ----------

t = resumo(df, "gender"); mostrar(t)
grafico_taxa(t, "gender", titulo_txt=f"Gênero não diferencia o churn: {t.taxa.min():.0%} vs {t.taxa.max():.0%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Dados faltantes
# MAGIC ### B4. Cliente com dado faltante tem perfil diferente?

# COMMAND ----------

t = resumo(df, "dado_faltante"); mostrar(t); grafico_taxa(t, "dado_faltante")

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Consumo
# MAGIC ### C3. Streaming reduz o churn? (não)

# COMMAND ----------

t = resumo(net, "streaming"); mostrar(t)
grafico_taxa(t, "streaming", titulo_txt=f"Streaming quase não muda o churn: de {t.taxa.min():.0%} a {t.taxa.max():.0%}",
             subtitulo="Taxa de churn por uso de streaming, só clientes com internet")

# COMMAND ----------

# MAGIC %md
# MAGIC ### C4. Ter mais produtos retém mais? (não de forma contínua)
# MAGIC Quem tem só 1 produto (em geral só telefone) quase não sai. O risco sobe com 2–5 produtos e só cai com 6+. Não dá para dizer "mais produtos retêm mais".

# COMMAND ----------

t = resumo(df, "faixa_produtos", ordem=["1", "2–3", "4–5", "6+"]); mostrar(t)
grafico_taxa(t, "faixa_produtos", ordenar=False,
             titulo_txt="Churn por quantidade de produtos sobe e depois cai: relação em U invertido",
             subtitulo="Taxa de churn por quantidade de produtos contratados")

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Preço
# MAGIC ### D2. Quem sai paga mais pelo mesmo pacote? (não)
# MAGIC Olhando só a quantidade de produtos, quem sai parece pagar mais. Mas isso é efeito da mistura fibra × DSL (fibra é mais cara e sai mais). Separando por tipo de internet, a diferença praticamente some.

# COMMAND ----------

d2 = (net.groupby(["InternetService", "qtd_produtos", "Churn"]).MonthlyCharges.mean().unstack())
d2 = d2.dropna()
d2["diferença (saiu - ficou)"] = d2["Yes"] - d2["No"]
display(d2.round(1))
dif_bruta = df.groupby(["qtd_produtos", "Churn"]).MonthlyCharges.mean().unstack().dropna()
print(f"Diferença média sem separar internet: {dinheiro((dif_bruta.Yes - dif_bruta.No).mean())}")
print(f"Diferença média separando internet:   {dinheiro(d2['diferença (saiu - ficou)'].mean())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### D3. A fatura pesa no orçamento? (não, repete a fatura)
# MAGIC O peso da fatura na renda parece ter efeito, mas dentro da mesma faixa de fatura ele não muda o churn.

# COMMAND ----------

t = resumo(df, "quintil_peso", ordem=["Q1 (pesa menos)", "Q2", "Q3", "Q4", "Q5 (pesa mais)"]); mostrar(t)
df["peso_renda"] = df.MonthlyCharges / df.MonthlyIncome
df["peso_na_faixa"] = (df.groupby("faixa_fatura", observed=True)["peso_renda"]
                         .transform(lambda s: pd.qcut(s, 3, labels=["baixo", "médio", "alto"]).astype(str)))
display(df.pivot_table(index="faixa_fatura", columns="peso_na_faixa", values="churn", aggfunc="mean", observed=True)
          .style.format("{:.0%}").background_gradient(cmap="Blues", axis=None))

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Pagamento
# MAGIC ### E2. Fatura digital está ligada a mais churn?

# COMMAND ----------

t = resumo(df, "PaperlessBilling"); mostrar(t); grafico_taxa(t, "PaperlessBilling")

# COMMAND ----------

# MAGIC %md
# MAGIC ## F. Sentimento do feedback (descritivo)
# MAGIC Score com **VADER** (modelo leve para inglês): de -1 (negativo) a +1 (positivo); negativo < -0,05 < neutro < +0,05 < positivo.
# MAGIC
# MAGIC **Fora do modelo e fora da síntese.** O sentimento herda o problema do texto (parte das respostas veio depois do cancelamento). E o VADER mede o tom das palavras, não a intenção: "satisfeito, mas troquei por uma oferta melhor" sai positivo.

# COMMAND ----------

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analisador = SentimentIntensityAnalyzer()

def sentimento(texto):
    return analisador.polarity_scores(texto if isinstance(texto, str) else "")["compound"]

ORDEM_SENT = ["Negativo", "Neutro", "Positivo"]
df["sentimento_score"] = df.CustomerFeedback.map(sentimento)
df["sentimento"] = pd.cut(df.sentimento_score, [-1.01, -0.05, 0.05, 1.01], labels=ORDEM_SENT)

display(df.groupby("Churn").sentimento_score.describe().round(2))
t_sent = resumo(df, "sentimento", ordem=ORDEM_SENT); mostrar(t_sent)
comp = t_sent.set_index("sentimento").pct_churners.reindex(ORDEM_SENT).fillna(0)

fig, ax = plt.subplots(figsize=(9, 3.2))
ax.barh(comp.index.astype(str), comp.values,
        color=[DESTAQUE if s == "Positivo" else CONTEXTO for s in comp.index], height=0.6)
for i, v in enumerate(comp.values):
    ax.text(v + 0.01, i, f"{v:.0%} dos churners", va="center", fontsize=10)
ax.set_xlim(0, 1); ax.xaxis.set_visible(False); ax.spines["bottom"].set_visible(False); ax.tick_params(left=False)
titulo(ax, f"{comp['Positivo']:.0%} dos clientes que saíram deixaram um feedback positivo",
       "Distribuição dos churners pelo sentimento do feedback: sair não é sinônimo de insatisfação")
plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Por que os satisfeitos saem?
# MAGIC Se os churners com feedback positivo citam preço e concorrência, eles saem por oferta, não por insatisfação.

# COMMAND ----------

ch = df[df.churn == 1]
x = ch.groupby("sentimento", observed=True)[list(COL_MOTIVO.values())].mean()
x.columns = list(COL_MOTIVO)
display(x.style.format("{:.0%}").background_gradient(cmap="Blues", axis=None))

print("Exemplos de churners com feedback positivo:\n")
for txt in ch[ch.sentimento == "Positivo"].CustomerFeedback.sample(3, random_state=1):
    print("-", txt[:300], "...\n")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Nuvem de palavras por sentimento (em português)
# MAGIC Os feedbacks estão em inglês; cada palavra é traduzida por um dicionário (`TRADUCAO`) e as variações se juntam. Só aparecem palavras do dicionário. Tamanho = frequência dentro do grupo.

# COMMAND ----------

from wordcloud import WordCloud
from collections import Counter

TRADUCAO = {
    "affordable": "acessível", "appreciate": "valorizo", "appreciated": "valorizo", "available": "disponível",
    "better": "melhor", "bill": "fatura", "bills": "fatura", "budget": "orçamento", "chose": "escolhi",
    "churning": "cancelar", "compared": "comparado", "competitive": "competitivo", "complaints": "reclamações",
    "connection": "conexão", "connectivity": "conexão", "consistent": "estável", "continue": "continuar",
    "convenience": "praticidade", "convenient": "prático", "cost": "custo", "deal": "oferta",
    "decision": "decisão", "different": "outra operadora", "difficult": "difícil", "disappointed": "decepcionado",
    "downtime": "instabilidade", "easy": "fácil", "elsewhere": "concorrente", "expectations": "expectativas",
    "expected": "esperado", "expensive": "caro", "experienced": "enfrentei", "extremely": "extremamente",
    "fast": "rápido", "flexibility": "flexibilidade", "free": "sem complicação", "frequent": "frequente",
    "frustrating": "frustrante", "great": "ótimo", "happy": "feliz", "hassle": "complicação", "high": "alto",
    "higher": "mais alto", "highly": "muito", "improve": "melhorar", "inconsistent": "instável",
    "inconvenient": "inconveniente", "interruptions": "interrupções", "issues": "problemas", "lack": "falta",
    "loyal": "fiel", "mail": "correio", "mailing": "correio", "meet": "atender", "methods": "formas de pagamento",
    "modern": "moderno", "never": "nunca", "offer": "oferta", "offered": "oferecido", "online": "online",
    "options": "opções", "outages": "quedas", "outdated": "ultrapassado", "paid": "pago", "payments": "pagamentos",
    "personal": "pessoal", "plans": "planos", "pleased": "satisfeito", "positive": "positivo", "prefer": "prefiro",
    "preferred": "prefiro", "price": "preço", "pricing": "preço", "provide": "oferecer", "quality": "qualidade",
    "reasonable": "razoável", "received": "recebi", "receiving": "recebi", "recommend": "recomendo",
    "reliability": "confiabilidade", "reliable": "confiável", "satisfied": "satisfeito", "slow": "lento",
    "speed": "velocidade", "speeds": "velocidade", "switched": "troquei", "unfortunately": "infelizmente",
    "unreliable": "instável", "wish": "gostaria", "worth": "vale a pena", "disappointing": "decepcionante",
    "poor": "ruim", "terrible": "péssimo", "rude": "grosseiro", "support": "suporte", "cheaper": "mais barato",
}

def frequencias_pt(textos):
    c = Counter()
    for txt in textos.fillna(""):
        c.update(TRADUCAO[w] for w in re.findall(r"[a-z]+", txt.lower()) if w in TRADUCAO)
    return c

GRUPOS_SENT = [("Negativo", DESTAQUE), ("Neutro", TEXTO_2), ("Positivo", TEXTO_2)]
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, (grupo, cor) in zip(axes, GRUPOS_SENT):
    sub = df[df.sentimento == grupo]
    nuvem = WordCloud(width=700, height=500, background_color=FUNDO, max_words=45, random_state=42,
                      color_func=lambda *a, cor=cor, **k: cor).generate_from_frequencies(frequencias_pt(sub.CustomerFeedback))
    ax.imshow(nuvem, interpolation="bilinear"); ax.axis("off")
    ax.set_title(f"{grupo}  ·  {len(sub):,} feedbacks  ·  churn {sub.churn.mean():.0%}", loc="left",
                 fontsize=12, fontweight="bold", color=DESTAQUE if grupo == "Negativo" else TEXTO_2)
fig.suptitle("O que os clientes dizem em cada grupo de sentimento", x=0.01, ha="left", fontsize=14, fontweight="bold", color=TEXTO)
plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Palavras que mais diferenciam quem sai
# MAGIC A nuvem mostra frequência mas não compara grupos. Aqui, a diferença no % de feedbacks que citam cada palavra entre quem saiu e quem ficou.

# COMMAND ----------

def presenca_pt(textos):
    c = Counter()
    for txt in textos.fillna(""):
        c.update({TRADUCAO[w] for w in re.findall(r"[a-z]+", txt.lower()) if w in TRADUCAO})
    return pd.Series(c) / len(textos)

pal = pd.DataFrame({"Saiu": presenca_pt(df.loc[df.churn == 1, "CustomerFeedback"]),
                    "Ficou": presenca_pt(df.loc[df.churn == 0, "CustomerFeedback"])}).fillna(0)
pal["dif"] = pal.Saiu - pal.Ficou
top = pd.concat([pal.nlargest(8, "dif"), pal.nsmallest(6, "dif")]).sort_values("dif")

fig, ax = plt.subplots(figsize=(9, 0.42 * len(top) + 1.6))
ax.barh(top.index, top.dif, color=[DESTAQUE if d > 0 else CONTEXTO for d in top.dif], height=0.6)
ax.axvline(0, color=TEXTO_2, lw=1)
for i, d in enumerate(top.dif):
    ax.text(d + (0.01 if d > 0 else -0.01), i, f"{d:+.0%}", va="center", ha="left" if d > 0 else "right", fontsize=9.5)
ax.xaxis.set_visible(False); ax.spines["bottom"].set_visible(False); ax.spines["left"].set_visible(False); ax.tick_params(left=False)
lim = top.dif.abs().max() * 1.3; ax.set_xlim(-lim, lim)
w = top.index[-1]
titulo(ax, f"'{w}' aparece em {pal.loc[w, 'Saiu']:.0%} dos feedbacks de quem saiu e {pal.loc[w, 'Ficou']:.0%} de quem ficou",
       "Diferença no % de feedbacks que citam a palavra: azul = mais usada por quem saiu")
plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pagamento × sentimento
# MAGIC Mesmo entre clientes com feedback positivo, o cheque eletrônico tem churn bem acima dos outros meios.

# COMMAND ----------

taxa_ps = pd.crosstab(df.PaymentMethod, df.sentimento, values=df.churn, aggfunc="mean").reindex(ORDEM_PAG)
n_ps = pd.crosstab(df.PaymentMethod, df.sentimento).reindex(ORDEM_PAG)
print("Taxa de churn por meio de pagamento e sentimento")
display(taxa_ps.style.format("{:.0%}").background_gradient(cmap="Blues", axis=None))
print("Número de clientes em cada célula")
display(n_ps)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Ficaram, mas estão insatisfeitos (só base histórica)
# MAGIC Lista qualitativa, só na base histórica.

# COMMAND ----------

risco_hist = df[(df.churn == 0) & (df.sentimento == "Negativo")]
print(f"{len(risco_hist)} clientes ficaram mas escreveram feedback negativo ({dinheiro(risco_hist.MonthlyCharges.sum())}/mês de receita)")
display(risco_hist[["customerID", "Contract", "InternetService", "PaymentMethod", "MonthlyCharges", "sentimento_score"]]
        .sort_values("sentimento_score").head(15))