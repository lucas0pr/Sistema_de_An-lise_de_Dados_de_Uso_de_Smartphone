# Sistema de Análise de Dados de Uso de Smartphone

Este projeto implementa um sistema completo para análise do dataset **"Smartphone Usage and Behavioral Dataset"** utilizando Python.

## Descrição do Dataset

O dataset contém informações sobre o comportamento de uso de smartphones por **1000 usuários**, incluindo:

* **User_ID**: Identificador único do usuário
* **Age**: Idade do usuário
* **Gender**: Gênero do usuário (Male/Female)
* **Total_App_Usage_Hours**: Tempo total de uso de apps em horas
* **Daily_Screen_Time_Hours**: Tempo diário de tela em horas
* **Number_of_Apps_Used**: Número de apps utilizados
* **Social_Media_Usage_Hours**: Tempo de uso de redes sociais em horas
* **Productivity_App_Usage_Hours**: Tempo de uso de apps de produtividade em horas
* **Gaming_App_Usage_Hours**: Tempo de uso de jogos em horas
* **Location**: Localização do usuário

---

## Funcionalidades Implementadas

### 1. Leitura do Dataset

* Carregamento do arquivo CSV utilizando **pandas**
* Conversão automática de tipos de dados
* Validação básica dos dados

### 2. Tratamento Inicial dos Dados

* **Remoção de valores nulos/inválidos**: Identificação e tratamento de registros com dados inconsistentes
* **Padronização de colunas**:

  * Conversão de gênero e localização para formato título (Male/Female, cidades)
  * Padronização de tipos numéricos
* **Verificação de duplicatas**: Identificação e remoção de registros duplicados
* **Detecção e tratamento de outliers**: Análise usando método IQR
* **Correção de inconsistências**: Ajuste do `Total_App_Usage_Hours` e `Daily_Screen_Time_Hours` quando necessário

### 3. Análise Exploratória

* **Estatísticas descritivas**: Média, mediana, mínimo, máximo e desvio padrão
* **Análise por gênero**: Distribuição e médias por categoria
* **Análise por localização**: Distribuição e comportamento por cidade
* **Análise de correlações**: Coeficiente de Pearson entre variáveis principais
* **Insights básicos**: Resumo das métricas mais importantes

### 4. Visualizações

* **Gráficos salvos em PNG** utilizando matplotlib/seaborn:

  * Distribuição por gênero
  * Distribuição por localização
  * Comparação de uso de apps por gênero
  * Evolução do número de apps vs idade
* Diretório de saída: `visualizations/`

### 5. Relatório de Insights

* **Resumo geral**: Visão geral do dataset
* **Insights demográficos**: Análise de gênero e localização
* **Insights de comportamento**: Padrões de uso por diferentes segmentos
* **Insights de correlação**: Relações entre variáveis
* **Insights por faixa etária**: Comportamento por grupos de idade
* **Recomendações**: Sugestões para análises futuras

---

## Estrutura do Projeto

```
├── mobile_usage_behavioral_analysis.csv  # Dataset original
├── smartphone_usage_analysis.py          # Script principal de análise
├── reports/                              # Diretório de relatórios
│   ├── visualizacoes_texto.txt          # Visualizações em texto
│   └── relatorio_completo.txt           # Relatório completo
└── visualizations/                       # Gráficos em PNG
└── README.md                             # Este arquivo
```

---

## Como Executar

### Pré-requisitos

* Python 3.6 ou superior
* Bibliotecas: `pandas`, `numpy`, `matplotlib`, `seaborn`

### Execução

```bash
python3 smartphone_usage_analysis.py
```

### Saída Esperada

1. **Análise no console**: Relatório completo impresso no terminal
2. **Arquivos de relatório**:

   * `reports/visualizacoes_texto.txt`
   * `reports/relatorio_completo.txt`
3. **Gráficos em PNG**: Salvos no diretório `visualizations/`

---

## Principais Insights do Dataset

### Demográficos

* **Distribuição de gênero**: 51.7% Masculino, 48.3% Feminino
* **Principais localizações**:

  * New York: 24.3%
  * Phoenix: 19.9%
  * Chicago: 19.2%
* **Faixa etária**: 18 a 59 anos, média: 38.7 anos

### Comportamento

* **Tempo médio de uso de apps**: 7.43 horas/dia
* **Tempo médio de tela diário**: 9.39 horas/dia
* **Número médio de apps usados**: 16.6 apps
* **Uso por categoria**:

  * Redes sociais: 2.46h/dia
  * Produtividade: 2.50h/dia
  * Jogos: 2.48h/dia

### Correlações

* **Total_App_Usage_Hours vs Daily_Screen_Time_Hours**: correlação positiva moderada (0.468)
* Baixa correlação entre idade e padrões de uso

### Padrões por Faixa Etária

* **18-25 anos**: 16.4 apps, 7.48h/dia
* **26-35 anos**: 17.0 apps, 7.31h/dia
* **36-45 anos**: 16.4 apps, 7.59h/dia
* **46-55 anos**: 16.7 apps, 7.28h/dia
* **56+ anos**: 17.0 apps, 7.60h/dia

---

## Recursos Técnicos

* **Python 3.x**
* **Pandas**: Manipulação de dados
* **Numpy**: Cálculos numéricos
* **Matplotlib e Seaborn**: Visualizações gráficas
* **Detecção de outliers**: Método IQR
* **Cálculo de correlação**: Coeficiente de Pearson

---

## Limitações e Melhorias Futuras

### Limitações Atuais

* Visualizações limitadas a gráficos estáticos
* Dataset estático (sem análise temporal)
* Análise limitada a 1000 registros

### Melhorias Possíveis

1. Gráficos interativos ou dashboards
2. Análise temporal com séries históricas
3. Machine Learning (clusterização ou previsão)
4. Mapas e análise geográfica
5. Conectar a dados em tempo real

---

## Conclusão

O sistema demonstra uma análise completa de dados de uso de smartphone, desde a carga e limpeza dos dados até a geração de insights acionáveis. Os resultados mostram padrões de comportamento, diferenças demográficas e correlações úteis para estratégias de desenvolvimento de apps e marketing digital.
