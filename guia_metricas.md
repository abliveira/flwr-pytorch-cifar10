# Guia de leitura das métricas do treinamento federado

Este guia explica as métricas registradas pela estratégia `LoggingFedAvg` em um experimento de aprendizado federado com Flower. O objetivo é documentar o significado dos campos e como interpretá-los, independentemente da quantidade de clientes, do número de rodadas ou dos resultados de uma execução específica.

**Importante:** todos os valores numéricos apresentados neste guia são exemplos fictícios, usados apenas para demonstrar a leitura dos registros. Eles não representam resultados esperados nem resultados de uma execução real.

## 1. Fluxo de uma rodada

Cada rodada do Flower normalmente segue esta sequência:

1. `configure_train`: o servidor seleciona os clientes que participarão do treinamento.
2. `aggregate_train`: os clientes treinam localmente e o servidor agrega as atualizações do modelo. O log pode apresentar a quantidade de exemplos e a perda de treinamento de cada cliente, além da perda agregada.
3. `configure_evaluate` e `aggregate_evaluate`: os clientes avaliam o modelo recebido em seus próprios conjuntos de validação. O log pode apresentar as métricas individuais e um resumo comparativo entre os clientes.
4. `Global evaluation`: quando configurada, o servidor avalia o modelo global em um conjunto de avaliação centralizado, separado dos dados locais usados pelos clientes.

A avaliação local dos clientes e a avaliação centralizada do servidor são feitas sobre conjuntos de dados potencialmente diferentes. Por isso, os resultados não precisam coincidir.

## 2. Métricas de treinamento local

| Campo no log | Significado | Como interpretar |
|---|---|---|
| `num-examples` / `EXAMPLES` | Número de exemplos usado pelo cliente no treinamento reportado | Ajuda a compreender o tamanho da contribuição local e, quando aplicável, o peso usado na agregação. |
| `train_loss` / `TRAIN LOSS` | Perda reportada após o treinamento local | Em geral, valores menores indicam menor erro segundo a função de perda. Analise junto com as métricas de validação. |
| `Weighted train loss` | Perda agregada dos clientes, ponderada pela quantidade de exemplos quando essa é a regra configurada | Resume a perda de treinamento levando em conta o tamanho das contribuições dos clientes. |
| `Replies` | Número de respostas recebidas com sucesso na etapa | Compare com o número de clientes selecionados para verificar a participação efetiva. |
| `Failures` | Número de falhas registradas na etapa | Falhas podem reduzir o número de clientes cujos resultados entram na agregação. |

Exemplo fictício de linha de treinamento:

```text
Client 01           1,200          1.2450
```

Nesta linha ilustrativa, `Client 01` reportou 1.200 exemplos e uma perda de treinamento de 1,2450. Os valores são apenas demonstrativos.

## 3. Métricas de validação individual dos clientes

| Campo no log | Significado | Como interpretar |
|---|---|---|
| `N` / `num-examples` | Quantidade de exemplos usados na validação do cliente | É importante ao comparar clientes com tamanhos de validação diferentes. |
| `LOSS` / `eval_loss` | Perda no conjunto de validação local | Menor geralmente é melhor, considerando a mesma função de perda e condições comparáveis. |
| `ACCURACY` / `eval_acc` | Proporção de previsões corretas | Maior geralmente é melhor. Pode ser apresentada como fração (`0.82`) ou porcentagem (`82%`). |
| `MACRO-P` / `eval_precision` | Precisão macro: média da precisão calculada por classe | Dá o mesmo peso a cada classe na média, em vez de dar mais peso às classes mais frequentes. |
| `MACRO-R` / `eval_recall` | Recall macro: média do recall calculado por classe | Indica a capacidade média de identificar exemplos pertencentes a cada classe. |
| `MACRO-F1` / `eval_f1` | F1 macro: média do F1 calculado por classe | Resume o equilíbrio entre precisão e recall, atribuindo o mesmo peso a cada classe. |
| `eval_confidence` | Média da maior probabilidade prevista para cada exemplo | Representa a confiança declarada pelo modelo, não a proporção de acertos. |
| `eval_correct_confidence` | Confiança média nas previsões corretas | Mostra a confiança do modelo nos exemplos que classificou corretamente. |
| `eval_wrong_confidence` | Confiança média nas previsões incorretas | Valores altos podem indicar que o modelo comete erros com alta confiança. |

Precisão, recall e F1 são chamados de **macro** quando a métrica é calculada separadamente por classe e, em seguida, é feita a média entre as classes. Confirme a implementação de `test_metrics` para verificar que esse é o método utilizado.

Exemplo fictício de tabela de validação:

| Cliente | N | LOSS | ACCURACY | MACRO-P | MACRO-R | MACRO-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Client 01 | 240 | 0,72 | 81,25% | 0,8120 | 0,8080 | 0,8090 |
| Client 02 | 300 | 0,85 | 78,00% | 0,7860 | 0,7790 | 0,7800 |

Os números acima são inventados apenas para ilustrar o formato. Nesse exemplo, o `Client 01` tem acurácia de validação maior, mas isso não permite concluir, por si só, que ele tenha dados mais fáceis ou que a distribuição seja IID.

## 4. Resumo comparativo entre clientes

A estratégia de logging pode apresentar estatísticas de acurácia calculadas a partir dos clientes que responderam à avaliação naquela rodada.

| Campo no log | Significado | Como interpretar |
|---|---|---|
| `Mean` | Média aritmética simples das acurácias dos clientes incluídos no resumo | Dá o mesmo peso a cada cliente, mesmo que os conjuntos de validação tenham tamanhos diferentes. |
| `Worst` | Menor acurácia observada entre os clientes incluídos | Mostra a acurácia mais baixa naquela rodada. |
| `Best` | Maior acurácia observada entre os clientes incluídos | Mostra a acurácia mais alta naquela rodada. |
| `Gap` | Diferença entre `Best` e `Worst` | Geralmente apresentada em pontos percentuais (pp); descreve a dispersão entre o melhor e o pior resultado observado. |
| `Replies` / `Failures` | Quantidade de respostas bem-sucedidas e de falhas | Dá contexto sobre quantos clientes contribuíram para o resumo. |
| `Client ID mapping` | Associação entre rótulos como `Client 01` e identificadores técnicos como `node_id` | Ajuda a acompanhar um cliente entre rodadas. O `node_id` não é uma métrica de desempenho. |

Exemplo fictício:

```text
Client accuracy (unweighted) | Mean: 79.60% | Worst: 78.00% | Best: 81.25% | Gap: 3.25 pp
```

Nesse exemplo, a média simples das acurácias é 79,60%, e a diferença entre 81,25% e 78,00% é 3,25 pontos percentuais. Esses valores são ilustrativos.

A média simples entre clientes (`Mean`) não precisa ser igual à acurácia agregada reportada pelo Flower. Quando a agregação é ponderada por `num-examples`, clientes com mais exemplos têm maior peso na métrica agregada; já a média simples dá o mesmo peso a cada cliente. As duas medidas coincidem quando os clientes têm o mesmo tamanho de conjunto e as mesmas métricas individuais são consideradas.

Um `Gap` pequeno indica que as acurácias observadas estão próximas naquela rodada, mas **não prova** que os dados dos clientes sejam IID. Um `Gap` alto também não prova, sozinho, que exista heterogeneidade: diferenças de tamanho de amostra e variabilidade estatística podem contribuir para a dispersão.

## 5. Métricas da avaliação global no servidor

Quando uma função de avaliação centralizada está configurada, o servidor calcula métricas do modelo global em um conjunto de dados de avaliação centralizado.

| Campo no log | Significado | Como interpretar |
|---|---|---|
| `loss` | Perda no conjunto centralizado | Menor geralmente é melhor, para a mesma função de perda e o mesmo conjunto de avaliação. |
| `accuracy` | Proporção de previsões corretas no conjunto centralizado | Maior geralmente é melhor. |
| `precision` | Precisão macro no conjunto centralizado, se assim definida pela função de métricas | Maior geralmente é melhor, interpretada junto com recall e F1. |
| `recall` | Recall macro no conjunto centralizado, se assim definida pela função de métricas | Maior geralmente é melhor. |
| `f1` | F1 macro no conjunto centralizado, se assim definida pela função de métricas | Ajuda a resumir o equilíbrio entre precisão e recall. |
| `confidence` | Confiança média das previsões | Não equivale à acurácia e não mede calibração diretamente. |
| `correct_confidence` | Confiança média nos exemplos classificados corretamente | Permite comparar a confiança nos acertos com a confiança nos erros. |
| `wrong_confidence` | Confiança média nos exemplos classificados incorretamente | Uma confiança elevada nos erros pode justificar uma análise de calibração. |

Exemplo fictício de saída resumida:

```text
ROUND 03 | GLOBAL EVALUATION
Loss: 0.92 | Accuracy: 78.40% | Macro precision: 0.7860 | Macro recall: 0.7840 | Macro-F1: 0.7810
```

Essa linha serve apenas para exemplificar como ler os campos; não define metas de desempenho.

Se o sistema registra uma avaliação antes do início do treinamento, ela pode aparecer como rodada inicial ou sob o índice `0` no resumo final. As avaliações seguintes costumam corresponder às rodadas de treinamento, mas a convenção exata depende da implementação e do formato do log.

## 6. Como ler o bloco `Final results`

O bloco `Final results` do Flower é um resumo de métricas registradas durante a execução; em geral, não representa uma nova avaliação independente.

- `Aggregated ClientApp-side Train Metrics`: métricas de treinamento agregadas por rodada, como `train_loss`.
- `Aggregated ClientApp-side Evaluate Metrics`: métricas de avaliação agregadas a partir das respostas dos clientes, como perda, acurácia, precisão, recall, F1 e confiança, conforme o que cada cliente reportou.
- `ServerApp-side Evaluate Metrics`: métricas calculadas pela avaliação centralizada do servidor, caso ela esteja configurada.

Os valores podem aparecer em notação científica. Por exemplo, `7.8400e-01` equivale a `0.784`, ou `78,4%` quando o campo representa uma proporção.

Os nomes exatos dos campos e a estrutura do resumo podem mudar conforme a versão do Flower e a implementação da estratégia. Use os nomes efetivamente presentes no log da execução.

## 7. Cuidados na interpretação

- **Conjuntos diferentes:** validação local e teste centralizado podem produzir resultados diferentes porque avaliam dados diferentes.
- **Métricas macro:** médias macro dão o mesmo peso a cada classe; não mostram, sozinhas, quais classes têm mais erros.
- **Confiança não é calibração:** um modelo pode emitir probabilidades elevadas e errar. A confiança média não demonstra se as probabilidades refletem frequências reais de acerto.
- **Perda e acurácia medem aspectos distintos:** a acurácia mede a proporção de previsões corretas; a perda também considera o grau de confiança atribuído às previsões, conforme a função utilizada.
- **Média entre clientes:** diferencie a média simples das métricas dos clientes da agregação ponderada por exemplos.
- **Clientes ausentes ou com falha:** o resumo pode refletir somente os clientes que responderam com sucesso. Consulte `Replies` e `Failures` ao comparar rodadas.
- **Média da perda por lotes:** se a implementação calcula a média simples das perdas dos lotes, um último lote menor pode receber o mesmo peso de um lote completo. Para obter uma média ponderada por exemplos, acumule a perda considerando o tamanho de cada lote.
- **IID e não-IID:** proximidade ou diferença entre métricas de clientes não comprova, por si só, o tipo de distribuição dos dados. A configuração da partição e a distribuição das classes também devem ser examinadas.
- **Calibração:** para avaliar a correspondência entre confiança e frequência real de acerto, considere métricas como Expected Calibration Error (ECE) e gráficos de confiabilidade.
- **Análise por classe:** matriz de confusão e métricas individuais por classe ajudam a localizar classes com desempenho baixo, algo que métricas agregadas podem ocultar.
