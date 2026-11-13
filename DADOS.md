# Entendimento dos Dados — Terra Signal (Previsão de Churn)

> Documento de referência da equipe (Grupo 2). A análise segue em etapas, e cada uma
> termina com as perguntas que ela levantou.
>
> **Convenção:**
> - Tudo que aparece sem marcação foi **observado diretamente nos arquivos**.
> - ❓ indica uma pergunta em aberto, escrita no ponto da análise em que ela surgiu.
> - Enquanto uma pergunta não for respondida, **nenhuma suposição sobre ela é tratada como fato**.
>
> **Status:** etapa de entendimento. Nenhuma limpeza foi aplicada e nenhum modelo foi treinado.

---

## Etapa 1 — Visão geral

A Terra Signal é uma empresa de telecomunicações (telefonia e internet). O objetivo do
hackathon é **prever se um cliente vai cancelar o serviço (churn)**.

| Arquivo | Linhas | Colunas | Papel |
|---|---|---|---|
| `history.csv` | 5.634 | 23 | Histórico com a resposta (`Churn`). |
| `inference.csv` | 1.409 | 22 | Mesmas colunas, **sem** `Churn`. Devemos prever `Churn` para cada linha. |

**O que observamos:**
- `Churn` no history: 4.130 `No` (73,3%) e 1.504 `Yes` (26,7%).
- Nenhum `customerID` se repete, nem dentro de um arquivo nem entre os dois.
- As proporções de `Contract`, `InternetService` e `PaymentMethod` são praticamente iguais
  nos dois arquivos (diferença ≤ 1,5 ponto percentual).
- A estrutura é idêntica à do dataset público *IBM Telco Customer Churn*, com duas colunas a
  mais: `CustomerFeedback` e `MonthlyIncome`. Por isso usamos esse dataset como referência
  para as descrições, mas sem considerar as definições dele como confirmadas.
- **Não existe nenhuma coluna de data.**

---

## Etapa 2 — Granularidade

**Cada linha representa um cliente** (um `customerID`), num retrato único do seu estado.

- Não é uma tabela de faturas, transações ou eventos: cada cliente aparece uma única vez,
  sem histórico ao longo do tempo.
- Os serviços contratados aparecem como colunas. Um cliente tem no máximo um tipo de
  contrato e um tipo de internet.
- Os valores de cobrança são do cliente como um todo: `MonthlyCharges` (mensal) e
  `TotalCharges` (acumulado).
- `Churn` é um rótulo único por cliente: saiu ou não saiu.

**Chave primária:** `customerID`, sempre no formato `NNNN-AAAAA` (4 dígitos, hífen e 5 letras
maiúsculas). Validado em 100% das 7.043 linhas.

### ❓ Perguntas desta etapa

- ❓ **`MonthlyCharges` é o valor do mês atual (no momento do retrato) ou uma média?** Isso
  muda a leitura da relação com `TotalCharges` (ver Etapa 4).
- ❓ **O que conta como churn: só cancelamento voluntário, ou também desligamento por
  inadimplência? Cancelar só um dos serviços conta?**

---

## Etapa 3 — Dicionário de dados

Legenda de tipo:
- **Nominal:** categorias sem ordem.
- **Ordinal:** categorias com ordem.
- **Discreta:** contagem ou valor inteiro.
- **Contínua:** medida numérica.
- **Data:** nenhuma variável deste conjunto é do tipo data.

### 3.1 Identificação e perfil do cliente

| Variável | Descrição | Tipo | Unidade | Valores válidos | ❓ Pergunta em aberto |
|---|---|---|---|---|---|
| `customerID` | Identificador único do cliente | Nominal (identificador) | — | Texto no padrão `NNNN-AAAAA` (ex.: `4578-PHJYZ`) | — |
| `gender` | Gênero do cliente | Nominal | — | `Female`, `Male` | — |
| `SeniorCitizen` | Indica se o cliente é idoso | Nominal (binária) | — | `0` = não, `1` = sim | — |
| `Partner` | Indica se o cliente tem cônjuge/companheiro(a) | Nominal (binária) | — | `Yes`, `No` | — |
| `Dependents` | Indica se o cliente tem dependentes | Nominal (binária) | — | `Yes`, `No` | — |
| `MonthlyIncome` | Renda mensal do cliente (pelo nome da coluna) | Contínua (registrada como inteiro) | Não informada | Inteiro de `100` a `9.993` | — |

### 3.2 Relacionamento e contrato

| Variável | Descrição | Tipo | Unidade | Valores válidos | ❓ Pergunta em aberto |
|---|---|---|---|---|---|
| `tenure` | Tempo de relacionamento do cliente com a empresa | Discreta | Meses (confirmado: ver nota A) | Inteiro de `0` a `72` | ❓ O que significa o valor `unknown` em 23 linhas? · ❓ Os clientes com `0` são novos? |
| `Contract` | Prazo do contrato vigente | **Ordinal** (fidelidade crescente) | — | `Month-to-month` < `One year` < `Two year` | — |
| `Churn` | **Variável-alvo:** o cliente cancelou? | Nominal (binária) | — | `Yes`, `No` (só em `history.csv`) | ❓ Inclui inadimplência? |

### 3.3 Serviços contratados

| Variável | Descrição | Tipo | Unidade | Valores válidos | ❓ Pergunta em aberto |
|---|---|---|---|---|---|
| `PhoneService` | Tem serviço de telefonia | Nominal (binária) | — | `Yes`, `No` | — |
| `MultipleLines` | Tem mais de uma linha telefônica | Nominal | — | `Yes`, `No`, `No phone service` | — |
| `InternetService` | Tipo de internet contratada | Nominal | — | `DSL`, `Fiber optic`, `No` | — |
| `OnlineSecurity` | Contratou segurança online (adicional) | Nominal | — | `Yes`, `No`, `No internet service` | ❓ `No` significa "tem internet e não contratou o adicional"? |
| `OnlineBackup` | Contratou backup online (adicional) | Nominal | — | `Yes`, `No`, `No internet service` | Mesma pergunta acima |
| `DeviceProtection` | Contratou proteção de dispositivo (adicional) | Nominal | — | `Yes`, `No`, `No internet service` | Mesma pergunta acima |
| `TechSupport` | Contratou suporte técnico (adicional) | Nominal | — | `Yes`, `No`, `No internet service` | Mesma pergunta acima |
| `StreamingTV` | Contratou streaming de TV | Nominal | — | `Yes`, `No`, `No internet service` | Mesma pergunta acima |
| `StreamingMovies` | Contratou streaming de filmes | Nominal | — | `Yes`, `No`, `No internet service` | Mesma pergunta acima |

### 3.4 Cobrança e pagamento

| Variável | Descrição | Tipo | Unidade | Valores válidos | ❓ Pergunta em aberto |
|---|---|---|---|---|---|
| `PaperlessBilling` | Recebe fatura digital (sem papel) | Nominal (binária) | — | `Yes`, `No` | — |
| `PaymentMethod` | Forma de pagamento | Nominal | — | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` | — |
| `MonthlyCharges` | Valor cobrado mensalmente | Contínua | Moeda ❓ | `18,25` a `118,75` (até 2 casas decimais) | ❓ Em que moeda (US$, R$)? Inclui impostos? |
| `TotalCharges` | Valor total cobrado ao longo do relacionamento | Contínua | Mesma moeda de `MonthlyCharges` ❓ | `18,80` a `8.684,80` (até 2 casas decimais) | ❓ Por que há 342 valores vazios? · ❓ O total inclui descontos, taxas ou mudanças de plano? |

### 3.5 Texto

| Variável | Descrição | Tipo | Unidade | Valores válidos | ❓ Pergunta em aberto |
|---|---|---|---|---|---|
| `CustomerFeedback` | Depoimento do cliente, em texto livre em inglês | Nominal (texto livre, não estruturado) | — | Texto de 49 a ~150 palavras (277 a 840 caracteres), sem duplicatas | ❓ Quando o feedback é coletado (antes ou depois do cancelamento)? Pode ser usado para prever? (ver Etapa 5.3) |

### Notas do dicionário

**A. Unidade de `tenure` (confirmada pelos dados).** Em 3.319 feedbacks o texto diz
*"for N months"*, e N bate com `tenure` em **100%** dos casos. Exemplo: `4226-KKDON` tem
`tenure = 8` e o feedback diz *"I have been a customer … for 8 months now"*.

**B. Faixa de `MonthlyCharges` por tipo de internet.** Os valores são coerentes com o serviço contratado:

| InternetService | Mínimo | Máximo |
|---|---|---|
| No | 18,25 | 26,90 |
| DSL | 23,45 | 94,80 |
| Fiber optic | 67,75 | 118,75 |

**C. Dependências entre colunas (100% consistentes).**
- `MultipleLines = No phone service` ⇔ `PhoneService = No`.
- As 6 colunas de serviços adicionais valem `No internet service` ⇔ `InternetService = No`.
- Não encontramos nenhuma linha que viole essas regras.

**D. `SeniorCitizen` é o único campo binário em `0`/`1`.** Todos os outros usam `Yes`/`No`.

---

## Etapa 4 — Qualidade dos dados

Fizemos uma checagem de nulos, valores fora do domínio, formatos e coerência entre colunas,
nos dois arquivos. **Nenhuma correção foi aplicada ainda.** Os tratamentos citados abaixo são
apenas possibilidades, que dependem das respostas.

### 4.1 `tenure` com o texto `unknown`

- **23 linhas** (19 history, 4 inference). Por causa delas, a coluna é lida como texto, não como número.
- Nas 19 linhas do history, 8 são churn (42%), contra 27% na média geral.
- Na maioria dos casos o valor poderia ser estimado por `TotalCharges ÷ MonthlyCharges` ou
  pelo feedback. Exemplo: `0135-NMXAP` dá 11,7 e o feedback diz *"customer for 12 months"*.
  Mas em `7301-ABVAD` a conta dá 73,3, acima do máximo de 72 que aparece no resto da base.

> ❓ **O que `unknown` significa: falha de extração, cliente migrado de outro sistema,
> informação não registrada? Podemos estimar o valor a partir de `TotalCharges ÷ MonthlyCharges`?**
>
> ❓ **72 meses é um teto real (por exemplo, a data de início do sistema) ou um valor truncado?**

### 4.2 `tenure = 0`

- **11 linhas** (10 history, 1 inference). Todas têm `TotalCharges` vazio.
- Os 10 do history são **todos `Churn = No`**.
- O feedback indica cliente recém-chegado. Exemplo: `2520-SGTTA` diz *"I recently signed up for a two-year contract…"*.

> ❓ **São clientes que entraram no mês do corte? Se sim, o churn deles chegou a ser
> observável, ou eles são sempre "No" por construção?**

### 4.3 `TotalCharges` vazio

- **342 linhas** (278 history, 64 inference). Há dois grupos:
  - **11** com `tenure = 0`: o vazio é coerente, porque o cliente ainda não foi cobrado.
  - **331** com `tenure > 0`: o vazio parece aleatório. Por exemplo, há 26 linhas com
    `tenure = 1` e 14 com `tenure = 72`. Exemplo: `8020-BWHYL` tem `tenure = 15`,
    `MonthlyCharges = 75,30` e `TotalCharges` vazio.
- O churn entre as linhas vazias é de 33%, contra 27% na média.

> ❓ **Para `tenure = 0`, o vazio significa "nada cobrado" (zero)?**
>
> ❓ **Para `tenure > 0`, o vazio é perda de dado na extração ou indica algo de negócio
> (fatura em aberto, inadimplência, contestação)?**

### 4.4 `TotalCharges` diferente de `tenure × MonthlyCharges`

- Em geral `TotalCharges ≈ tenure × MonthlyCharges`: a mediana da razão é 1,00 e o desvio-padrão é 0,05.
- Mas **56 linhas** divergem mais de 20%, com razão entre 0,69 e 1,49.
  Exemplo: `3345-JHUEO` tem 4 meses × 20,20 = 80,80 esperado, mas `TotalCharges = 55,70`.
- Nenhuma linha tem `TotalCharges` menor que `MonthlyCharges` (com `tenure > 0`).

> ❓ **`MonthlyCharges` é o valor atual, e o cliente pode ter mudado de plano? O total inclui
> descontos, taxas de instalação, multas ou meses proporcionais?**

### 4.5 `PhoneService` em minúsculas

- Além de `Yes` (6.318) e `No` (632), aparecem `yes` (43) e `no` (50). É a única coluna com esse problema.
- A taxa de churn é parecida: `Yes` 26,8%, `yes` 26,7%, `No` 25,8%, `no` 20,0% (n = 40, amostra pequena).

> ❓ **É só um erro de padronização, ou indica outro sistema de origem com outra regra?**

### 4.6 `MonthlyIncome` com comportamento estranho

- A distribuição é **quase perfeitamente uniforme** entre 100 e 9.993: cada decil tem entre
  638 e 749 clientes. Renda real costuma ser assimétrica, com muita gente na faixa baixa e
  poucos na alta.
- Correlação com `MonthlyCharges` ≈ 0,01.
- 244 clientes têm renda abaixo de 500.
- Em `6445-TNRXS` a renda (113) é **menor que a própria mensalidade** (114,70).

> ❓ **De onde vem essa renda (declarada pelo cliente, estimada por um bureau de crédito,
> gerada sinteticamente)? É individual ou familiar? Em que moeda?**

### 4.7 Resumo

| Problema | history | inference | Pergunta |
|---|---|---|---|
| `tenure = "unknown"` | 19 | 4 | O que significa `unknown`? Podemos estimar o valor? |
| `tenure = 0` | 10 | 1 | São clientes novos? O churn deles era observável? |
| `TotalCharges` vazio | 278 | 64 | Vazio é zero, perda de dado ou um sinal de negócio? |
| `TotalCharges` ≠ `tenure × MonthlyCharges` (> 20%) | 56 no total | | Mudança de plano? Descontos? |
| `PhoneService` em minúsculas | 70 | 23 | Mesmo significado? |
| `MonthlyIncome` uniforme e incoerente | toda a coluna | | Qual é a fonte e a unidade? |

Nenhum outro campo tem nulos, valores fora do domínio ou IDs duplicados.

---

## Etapa 5 — Relação descritiva com o churn (history)

Aqui olhamos só **taxas e médias** de churn em cada grupo. São descrições, não um modelo.
Nesta etapa as linhas com problemas da Etapa 4 foram ignoradas apenas nos cálculos numéricos.

### 5.1 Variáveis categóricas: taxa de churn por categoria

| Variável | Taxa de churn por categoria (n) | Diferença |
|---|---|---|
| `Contract` | Month-to-month **43%** (3.099) · One year 12% (1.188) · Two year **3%** (1.347) | 🔴 Muito grande |
| `InternetService` | Fiber optic **42%** (2.477) · DSL 19% (1.924) · No **7%** (1.233) | 🔴 Muito grande |
| `PaymentMethod` | Electronic check **45%** (1.888) · Mailed check 19% (1.283) · Bank transfer 17% (1.229) · Credit card 16% (1.234) | 🔴 Muito grande |
| `OnlineSecurity` | No 42% · Yes 15% · No internet 7% | 🟠 Grande |
| `TechSupport` | No 42% · Yes 15% · No internet 7% | 🟠 Grande |
| `OnlineBackup` | No 40% · Yes 22% · No internet 7% | 🟡 Moderada |
| `DeviceProtection` | No 39% · Yes 23% · No internet 7% | 🟡 Moderada |
| `SeniorCitizen` | 1 = 42% (928) · 0 = 24% (4.706) | 🟡 Moderada |
| `PaperlessBilling` | Yes 34% (3.327) · No 17% (2.307) | 🟡 Moderada |
| `Partner` | No 33% (2.907) · Yes 20% (2.727) | 🟡 Moderada |
| `Dependents` | No 32% (3.949) · Yes 15% (1.685) | 🟡 Moderada |
| `StreamingTV` / `StreamingMovies` | No ~33% · Yes ~30% · No internet 7% | ⚪ Pequena (a diferença vem de ter ou não internet) |
| `MultipleLines` | Yes 29% · No 25% · No phone 25% | ⚪ Pequena |
| `PhoneService` | Yes 27% · No 25% | ⚪ Nenhuma |
| `gender` | Female 27% · Male 26% | ⚪ Nenhuma |

**Leitura:** o churn se concentra em quem tem contrato mensal, internet por fibra, paga com
cheque eletrônico e não tem os serviços de segurança e suporte. A fibra ter churn alto
chama atenção, porque é o serviço mais caro e, em tese, o melhor.

### 5.2 Variáveis numéricas

| Variável | Média de quem ficou | Média de quem saiu | Observação |
|---|---|---|---|
| `tenure` | 37,4 meses (mediana 37) | 18,3 meses (mediana 10) | Quem sai tem bem menos tempo de casa |
| `MonthlyCharges` | 61,0 | 74,6 | Quem sai paga mais (ligado à fibra) |
| `MonthlyIncome` | 5.082 | 5.168 | Praticamente igual |

Churn por faixa de `tenure`:

| Meses | 0 | 1–6 | 7–12 | 13–24 | 25–48 | 49–72 |
|---|---|---|---|---|---|---|
| Clientes | 10 | 1.166 | 575 | 805 | 1.289 | 1.770 |
| Churn | 0% | **53%** | 35% | 28% | 22% | **9,5%** |

Churn por quintil de `MonthlyIncome`: 26% · 25% · 28% · 28% · 27%. Não muda com a renda.
Isso reforça a dúvida da Etapa 4.6.

### 5.3 `CustomerFeedback`: o texto parece conter a resposta

**O texto foi gerado a partir da própria linha:**
- Em **100%** dos 6.343 textos que citam um valor em `$`, o valor é exatamente o `MonthlyCharges` da linha.
- Em **100%** dos 3.319 textos que citam "N months", N é exatamente o `tenure` da linha.
- O tipo de internet citado no texto nunca contradiz `InternetService`.

**E o texto indica se o cliente saiu:**

| Termo no feedback | Ocorrências (history) | Taxa de churn |
|---|---|---|
| "decided" | 1.350 | **100%** |
| "better deal" | 404 | **100%** |
| "unfortunately" | 544 | 99,6% |
| "disappoint" | 204 | 98,5% |
| "cancel" | 897 | 92,5% |
| "happy" | 1.287 | 0,7% |
| "no plans" | 1.256 | **0%** |

**Atenção:** a palavra "churn" sozinha **não** indica saída. Ela aparece em 1.578 textos de
clientes que ficaram, em frases como *"…would make me want to cancel my service (churn)"*.

**Alguns textos estão no passado**, ou seja, foram escritos depois do cancelamento. Exemplo:
`0135-NMXAP` diz *"I recently cancelled my service with this company…"*.

O mesmo padrão aparece em `inference.csv`: 327 dos 1.409 textos contêm "decided".

> ❓ **Quando o feedback é coletado: antes do churn (pesquisa periódica) ou depois (pesquisa
> de cancelamento)?**
>
> ❓ **O texto é real ou sintético? Se for sintético, foi gerado já sabendo o rótulo `Churn`?**
>
> ❓ **É permitido usar `CustomerFeedback` para prever o churn? Num cenário real, esse texto
> estaria disponível antes do cliente sair?**

---

## Etapa 6 — Uso da previsão, custo dos erros e público da apresentação

Antes de limpar os dados e construir qualquer modelo, precisamos entender **para que** a
previsão serve e **quem** vai usá-la. As respostas mudam como o modelo deve ser ajustado e
como o resultado deve ser entregue.

### 6.1 Os dois tipos de erro, em linguagem simples

Nenhuma previsão acerta sempre. Quando o modelo errar, ele vai errar de um destes dois jeitos:

| | O cliente **ficou** de verdade | O cliente **saiu** de verdade |
|---|---|---|
| **O modelo disse "vai sair"** | ⚠️ **Alarme falso** (*falso positivo*) | ✅ Acerto: deu tempo de agir |
| **O modelo disse "vai ficar"** | ✅ Acerto: não precisava agir | 🚨 **Saída sem aviso** (*falso negativo*) |

**Alarme falso: o modelo avisa que o cliente vai sair, mas ele ia ficar.**
A empresa gasta à toa com uma ação de retenção (uma ligação, um desconto, um brinde) para
alguém que não ia embora. Pode até incomodar o cliente, ou acostumá-lo a pedir desconto.

**Saída sem aviso: o modelo diz que está tudo bem, mas o cliente cancela.**
A empresa perde o cliente e a receita dele sem ter tido chance de reagir. No histórico, quem
saiu pagava em média **74,60 por mês**.

**Uma comparação:** é como um alarme de incêndio. Se ele for muito sensível, toca toda vez que
alguém frita um ovo (alarme falso). Se for pouco sensível, pode ficar quieto durante um
incêndio de verdade (saída sem aviso). **Diminuir um tipo de erro quase sempre aumenta o
outro.** Por isso precisamos saber qual dos dois incomoda mais o negócio.

**Para ter uma noção de escala:** no histórico, de cada 100 clientes, cerca de 27 saíram.
Se a proporção em `inference.csv` for parecida (o que não está confirmado), seriam cerca de
376 dos 1.409 clientes.

> ❓ **Qual erro é pior para a Terra Signal: gastar com um cliente que não ia sair (alarme
> falso) ou perder um cliente sem ter sido avisado (saída sem aviso)? Quanto custa, em média,
> uma ação de retenção, e quanto vale um cliente que fica?**
>
> ❓ **Existe um limite de quantos clientes a equipe consegue contatar por mês?** Se a equipe só
> consegue ligar para 100 pessoas, por exemplo, o mais importante é acertar quem são os 100
> clientes de maior risco.

### 6.2 Quem vai usar a previsão e de que forma

A mesma previsão pode ser entregue de jeitos muito diferentes:
- uma **lista simples** de clientes com "vai sair" ou "vai ficar";
- uma **classificação de risco** (alto, médio, baixo) para priorizar o contato;
- um **painel** com o motivo provável de cada cliente (por exemplo, "contrato mensal + pouco tempo de casa");
- um **arquivo técnico** com probabilidades, para um time de dados integrar a outros sistemas.

Para uma pessoa da área de atendimento, uma probabilidade como "0,73" não diz muito. Já
"risco alto: contrato mensal e cliente há só 3 meses" ajuda a decidir o que fazer.

> ❓ **Quem vai usar o resultado no dia a dia: um time de dados experiente, um gestor ou
> analista de negócio, ou a equipe de atendimento e retenção sem formação técnica?**
>
> ❓ **Basta um "sim/não", ou a pessoa precisa de um nível de risco e do motivo de cada
> previsão para saber como abordar o cliente?**
>
> ❓ **Como e com que frequência a previsão será usada?** Em que ferramenta (planilha, painel,
> sistema de atendimento), com que frequência (todo mês? todo dia?) e qual ação vem depois
> (ligação, oferta de desconto, troca de plano)?

### 6.3 Para quem vamos apresentar o trabalho

Quem usa a previsão no dia a dia (6.2) não é necessariamente quem vai assistir à apresentação
do hackathon. O público da apresentação muda **o que mostrar e como explicar**:

| | Público técnico (dados, engenharia) | Público de negócio (gestão, produto, atendimento) |
|---|---|---|
| **O que interessa** | Como chegamos ao resultado e se ele é confiável | O que muda para a empresa e o que fazer com isso |
| **Como medir o resultado** | Métricas (acurácia, F1, ROC-AUC), validação | Clientes que dá para reter, receita protegida, custo das ações |
| **Os dois tipos de erro (6.1)** | Matriz de confusão, escolha do ponto de corte | "De cada 100 clientes que avisamos, quantos iam mesmo sair?" |
| **Problemas nos dados (Etapa 4)** | Detalhes da limpeza e das decisões tomadas | Só o que afeta a confiança no resultado |
| **O texto do feedback (Etapa 5.3)** | Discussão técnica sobre vazamento do alvo | "O texto já conta se o cliente saiu, então não ajuda a prever com antecedência" |
| **Formato** | Notebook, código, gráficos detalhados | Poucos slides, números simples, exemplos de clientes reais |

Se o público for misto, a estratégia mais comum é **começar pelo negócio** (o problema, o
impacto e a recomendação) e deixar os detalhes técnicos para o final ou para um apêndice.

> ❓ **Quem vai assistir e avaliar a apresentação: pessoas técnicas (dados, engenharia),
> pessoas que dominam só a lógica de negócio, ou um grupo misto?**
>
> ❓ **Quanto tempo temos para apresentar, e existe um formato obrigatório (slides, demonstração
> do notebook, painel)?**
>
> ❓ **Quais critérios a banca usa para avaliar: só o acerto da previsão, ou também a clareza,
> a utilidade para o negócio e a qualidade da análise?**

---

## Onde estamos

| Etapa | Situação |
|---|---|
| 1. Visão geral | ✅ Concluída |
| 2. Granularidade | ✅ Concluída |
| 3. Dicionário de dados | ✅ Concluído (com perguntas em aberto) |
| 4. Qualidade dos dados | ✅ Problemas mapeados. ⏸️ Correções aguardam respostas |
| 5. Relação descritiva com o churn | ✅ Concluída |
| 6. Uso da previsão, custo dos erros e público da apresentação | ✅ Perguntas levantadas. ⏸️ Aguardam respostas |
| 7. Limpeza e preparação | ⏳ Próxima etapa |
| 8. Modelagem | ⏳ Depois da limpeza. Depende das respostas sobre o uso do feedback (Etapa 5.3) e o uso da previsão (Etapa 6) |
