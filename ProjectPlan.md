# Project Plan

## Contents

- [Etapas](#etapas)
    1. [Infraestrutura e Setup](#0-infraestrutura-e-setup)
    1. [Dados](#1-dados)
    2. [Modelagem](#2-modelagem)
    3. [Entrega Técnica](#3-entrega-técnica)
    4. [Apresentação](#4-apresentação)
    5. [Submissão](#5-submissão)

- Discussões

---

## Etapas

### 0. Infraestrutura e Setup

- [x] Fork do repositório + colaboradores no workspace #infra
- [x] Preencher README com nome da equipe e integrantes #infra (due 11/07)
- [x] Validar acesso ao Databricks Free Edition (notebook, catálogo, MLflow) #infra (due 11/07)
- [x] Definir convenção de branches (ex: feat/eda, feat/model, docs/slides) #infra (due 11/07)
- [x] Configurar .gitignore (ignorar *.csv grandes, mlruns/, .ipynb_checkpoints/) #infra (due 11/07)

### 1. Dados

- [x] Carregar `history.csv` e `inference.csv `no workspace #data (due 11/07)
- [ ] EDA inicial: shape, tipos, missing, distribuição do target Churn #data (due 11/07)
- [ ] Tratar TotalCharges (valores em branco/string) #data (due 11/07)
- [ ] Tratar categóricas (One-Hot ou Target Encoding) #data (due 11/08)
- [ ] Processar CustomerFeedback (TF-IDF + SVD ou features simples) #data (due 11/08)
- [ ] Split treino/validação estratificado + pipeline reprodutível (sklearn Pipeline) #data (due 11/08)

### 2. Modelagem

- [ ] Baseline: Logistic Regression + métricas (AUC, Recall, F1) #model (due 11/07)
- [ ] Modelo principal: LightGBM (ou XGBoost CPU) com CV estratificada #model (due 11/08)
- [ ] Ajuste de threshold focado em Recall (justificar custo de falso negativo)  #model (due 11/08)
- [ ] Registrar experimentos no MLflow (parâmetros, métricas, artefato do modelo)  #model (due 11/08)
- [ ] Análise de feature importance + interpretação de negócio  #model (due 11/09)
- [ ] (nice-to-have) Comparar com modelo alternativo (ex: Random Forest) #model nice-to-have (due 11/10)

### 3. Entrega Técnica

- [ ] Gerar prediction.csv a partir de `inference.csv` #delivery (due 11/09)
- [ ] Validar formato do prediction.csv (colunas esperadas, sem NaN, probabilidades em [0,1]) #delivery (due 11/09)
- [ ] Subir `prediction.csv` no repo (ou link no README) #delivery (due 11/09)
- [ ] Construir Databricks App (Streamlit/Gradio): input de features → output probabilidade + sugestão de retenção #delivery (due 11/10)
- [ ] Testar App no Free Edition (confirmar que sobe e responde) #delivery (due 11/10)
- [ ] (fallback) Notebook com widgets interativos caso o App falhe #delivery (due 11/10)
- [ ] README final: arquitetura, como rodar, decisões técnicas #delivery (due 11/12)
- [ ] (nice-to-have) API com Model Serving #delivery nice-to-have (só se sobrar tempo)
- [ ] (nice-to-have) Chatbot com Genie/Custom Agents #delivery nice-to-have (só se sobrar tempo)

### 4. Apresentação

- [ ] Definir narrativa: problema → dados → modelo → produto → impacto de negócio #presentation (due 11/11)
- [ ] Montar slides (15 min, foco em negócio para banca mista) #presentation (due 11/11)
- [ ] Preparar demo do App (roteiro de 2-3 min) #presentation (due 11/11)
- [ ] Preparar respostas para perguntas prováveis (por que não Model Serving? por que Recall? como o time de vendas usa?) #presentation (due 11/11)
- [ ] Ensaio cronometrado (15 min) #presentation (due 11/12)
- [ ] Gravar vídeo backup da demo (caso o App caia na hora) #presentation (due 11/12)

### 5. Submissão

- [ ] Revisar estrutura do repo (pastas, README, notebooks limpos) #infra (due 11/12)
- [ ] Garantir todos os commits antes de 13/11 02:00 #infra (due 11/12)
- [ ] Preencher Google Forms com link do repo + `prediction.csv` #infra (due 11/12)
- [ ] Confirmar submissão (print/screenshot do Forms enviado) #infra (due 11/12)


---

## Discussões

