# Dúvidas sobre os Dados — Terra Signal

## Definição de churn

1. O que conta como churn: só cancelamento voluntário, ou também desligamento por inadimplência?
2. Cancelar só um dos serviços (por exemplo, só a internet) conta como churn?

## Tempo de relacionamento (`tenure`)

3. O que significa o valor `unknown` em 23 linhas: falha de extração, cliente migrado de outro sistema ou informação não registrada?
4. Podemos estimar o `tenure` desses clientes a partir de `TotalCharges ÷ MonthlyCharges`?
5. 72 meses é um teto real (por exemplo, a data de início do sistema) ou um valor truncado?
6. Os clientes com `tenure = 0` entraram no mês do corte?
7. Se sim, o churn deles chegou a ser observável, ou eles são sempre "No" por construção?

## Cobrança (`MonthlyCharges` e `TotalCharges`)

8. `MonthlyCharges` é o valor do mês atual (no momento do retrato) ou uma média?
9. Em que moeda estão `MonthlyCharges` e `TotalCharges` (US$, R$)? Os valores incluem impostos?
10. Para clientes com `tenure = 0`, o `TotalCharges` vazio significa "nada cobrado" (zero)?
11. Para clientes com `tenure > 0`, o `TotalCharges` vazio é perda de dado na extração ou indica algo de negócio (fatura em aberto, inadimplência, contestação)?
12. O cliente pode ter mudado de plano ao longo do tempo? O `TotalCharges` inclui descontos, taxas de instalação, multas ou meses proporcionais?

## Serviços contratados

13. `PhoneService` com `yes`/`no` em minúsculas é só um erro de padronização, ou indica outro sistema de origem com outra regra?
14. Nas colunas de serviços adicionais, `No` significa "tem internet e não contratou o adicional"?

## Renda (`MonthlyIncome`)

15. De onde vem a renda: declarada pelo cliente, estimada por um bureau de crédito ou gerada sinteticamente?
16. É renda individual ou familiar? Em que moeda?

## Feedback do cliente (`CustomerFeedback`)

17. Quando o feedback é coletado: antes do churn (pesquisa periódica), durante o pedido de cancelamento ou depois do cancelamento?
18. O texto é real ou sintético? Se for sintético, foi gerado já sabendo se o cliente saiu?
19. É permitido usar `CustomerFeedback` para prever o churn?
20. Num cenário real, esse texto estaria disponível antes do cliente sair?

## Custo dos erros

21. Qual erro é pior para a Terra Signal: gastar com um cliente que não ia sair (alarme falso) ou perder um cliente sem ter sido avisado (saída sem aviso)?
22. Quanto custa, em média, uma ação de retenção (ligação, desconto, brinde)?
23. Quanto vale, em média, um cliente que fica?
24. Existe um limite de quantos clientes a equipe consegue contatar por mês?

## Uso da previsão

25. Quem vai usar o resultado no dia a dia: um time de dados experiente, um gestor ou analista de negócio, ou a equipe de atendimento e retenção sem formação técnica?
26. Basta um "sim/não", ou a pessoa precisa de um nível de risco e do motivo de cada previsão para saber como abordar o cliente?
27. Em que ferramenta o resultado vai aparecer (planilha, painel, sistema de atendimento)?
28. Com que frequência a previsão será usada (uma vez, todo mês, todo dia)?
29. Qual ação vem depois que um cliente é apontado (ligação, oferta de desconto, troca de plano)?

## Apresentação

30. Quem vai assistir e avaliar a apresentação: pessoas técnicas (dados, engenharia), pessoas que dominam só a lógica de negócio, ou um grupo misto?
31. Quanto tempo temos para apresentar?
32. Existe um formato obrigatório (slides, demonstração do notebook, painel)?
33. Quais critérios a banca usa para avaliar: só o acerto da previsão, ou também a clareza, a utilidade para o negócio e a qualidade da análise?
