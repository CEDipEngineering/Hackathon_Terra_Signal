# App de atendimento preventivo

Aplicação Streamlit de uma tela para consultar a carteira `workspace.default.clientes_risco_churn` por `customerID` e registrar o resultado do atendimento em `workspace.default.atendimentos_churn`.

## O que a tela mostra

- score de risco de churn, tratado como priorização e não como probabilidade perfeitamente calibrada;
- internet, tenure, contrato, pagamento, suporte técnico e cobrança mensal;
- flags Top 10%, Top 20% e Top 30%, que são cumulativas;
- segmento e ação sugerida;
- formulário para registrar a interação do atendente.

`MonthlyIncome` não aparece no app por ser uma fonte externa com precisão não garantida.

## Preparação única

1. Execute `sql/criar_tabela_atendimentos_churn.sql` no Databricks.
2. Em **Databricks Apps**, crie um app customizado e conecte esta pasta do Git.
3. Adicione um recurso SQL Warehouse com a chave `sql-warehouse`.
4. Dê à identidade de serviço do app:
   - `SELECT` em `workspace.default.clientes_risco_churn`;
   - `MODIFY` em `workspace.default.atendimentos_churn`;
   - `CAN USE` no SQL Warehouse.
5. Faça o deploy e teste a busca por um `customerID` da tabela de scores.

O app usa autenticação da identidade de serviço do Databricks Apps. Não use token pessoal no código.
