-- Execute uma única vez no Databricks SQL ou em um notebook antes do deploy do app.
-- O app apenas inclui novos registros; ele não cria nem sobrescreve esta tabela.

CREATE TABLE IF NOT EXISTS workspace.default.atendimentos_churn (
    data_atendimento TIMESTAMP NOT NULL,
    customerID STRING NOT NULL,
    score_churn DOUBLE NOT NULL,
    top_faixa STRING,
    faixa_risco STRING,
    segmento_acao STRING,
    acao_sugerida STRING,
    motivo_atendimento STRING NOT NULL,
    recebeu_acao_promocional_recente STRING,
    desfecho_chamado STRING NOT NULL,
    avaliacao_servico STRING,
    avaliacao_atendimento STRING,
    sugestao_cliente STRING
)
USING DELTA;
