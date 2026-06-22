# Federated Demand Forecasting — Documentação Técnica

Dez lojas treinam seus próprios modelos localmente. Apenas os parâmetros aprendidos trafegam pela rede — nunca os dados brutos de vendas. O servidor central agrega os modelos e devolve um estado global melhorado a cada rodada.

---

## Índice

1. [Arquitetura do Sistema](#1-arquitetura-do-sistema)
2. [Pipeline MLOps](#2-pipeline-mlops)
3. [dataset.py — Pré-processamento](#3-datasetpy--pré-processamento)
4. [XGBoost — Treinamento e Agregação](#4-xgboost--treinamento-e-agregação)
5. [Croston — Modelo Estatístico e Agregação](#5-croston--modelo-estatístico-e-agregação)
   5b. [Croston Ensemble — Independência Estatística](#5b-croston-ensemble--independência-estatística)
6. [Baselines Centralizados e Modo Produto Único](#6-baselines-centralizados)
7. [Validade do Pipeline de Avaliação](#7-validade-do-pipeline-de-avaliação)
8. [Guia de Interpretação das Métricas](#8-guia-de-interpretação-das-métricas)
9. [XGBoost vs Croston — Comparação](#9-xgboost-vs-croston--comparação)
10. [Como Executar](#10-como-executar)

---

## 1. Arquitetura do Sistema

O sistema segue o padrão de **federated learning horizontal**: cada cliente possui dados de uma loja distinta com a mesma estrutura de features. O servidor nunca acessa dados brutos — apenas os parâmetros dos modelos.

```
             SERVIDOR FLOWER (Agregação Global)
                  num_rounds=10 · gRPC
           ↑ parâmetros locais  ↓ modelo global

  CA_1  CA_2  CA_3  CA_4  TX_1  TX_2  TX_3  WI_1  WI_2  WI_3
  (10 clientes — um por loja)
```

### Estrutura de Arquivos

```
dataset.py                  # load_data(), load_timeseries(), load_data_item(), load_timeseries_item(), …
main.py                     # FlowerXGBoostClient · XgbEnsembleStrategy
croston_model.py            # CrostonForecaster
croston_main.py             # FlowerCrostonClient · CrostonFedAvgStrategy (FedAvg clássico)
croston_ensemble_main.py    # FlowerCrostonEnsembleClient · CrostonEnsembleStrategy (sem agregação)
centralized.py              # Baselines centralizados + modo produto único (--item / --store)
```

---

## 2. Pipeline MLOps

Fluxo completo da leitura dos CSVs até a métrica agregada no servidor:

| Etapa | O que acontece |
|---|---|
| **Entrada** | Leitura de `sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv` com cache de módulo |
| **Limpeza** | Forward-fill → back-fill → fill(0) para NaN; clipping por IQR para outliers |
| **Features** | Codificação cíclica (sin/cos) de variáveis de calendário; preço médio semanal; flags SNAP e evento |
| **Normalização** | `StandardScaler` ajustado **apenas no treino** para evitar data leakage |
| **Split** | 80% treino / 20% teste respeitando a ordem cronológica (sem embaralhamento antes do split) |
| **Formato** | XGBoost → janela deslizante 28 dias × 12 features = 336 dims. Croston → série bruta 1-D |
| **Treino FL** | Cada cliente treina localmente; apenas parâmetros trafegam ao servidor |
| **Métricas** | MSE e MAE agregados com média ponderada pelo número de amostras de cada loja |

---

## 3. dataset.py — Pré-processamento

### Funções Públicas

| Função | Retorno | Usada por |
|---|---|---|
| `load_data(partition_id, *, return_scaler=False)` | `(X_train, y_train, X_test, y_test[, scaler])` | `main.py` (federado) |
| `load_data_torch(partition_id, num_partitions)` | `(train_loader, test_loader)` DataLoaders PyTorch | Modelos baseados em PyTorch |
| `load_timeseries(partition_id, num_partitions)` | `(y_train, y_test)` como `np.ndarray` 1-D | `croston_main.py`, `croston_ensemble_main.py` |
| `load_all_data()` | `(X_train, y_train, X_test, y_test, train_sizes, test_sizes, scalers)` | `centralized.py` (XGBoost) |
| `load_all_timeseries()` | `(y_trains, y_tests)` — listas com uma série por loja | `centralized.py` (Croston) |
| `load_data_item(item_id, store_id, *, return_scaler=False)` | `(X_train, y_train, X_test, y_test[, scaler])` | `main.py` (modo `ITEM_ID`), `centralized.py` (`--item`) |
| `load_timeseries_item(item_id, store_id)` | `(y_train, y_test)` como `np.ndarray` 1-D | `croston_main.py`, `croston_ensemble_main.py` (modo `ITEM_ID`), `centralized.py` (`--item`) |

> **Por que funções separadas para centralizado e federado?** As funções federadas (`load_data`, `load_timeseries`) usam um `partition_id` que mapeia para uma loja específica — essa interface é projetada para o Flower. As funções centralizadas (`load_all_data`, `load_all_timeseries`) carregam todas as lojas de uma vez, sem o conceito de partição, e retornam informações adicionais (tamanhos por loja, scalers) necessárias para computar métricas na escala bruta de vendas.

> **Por que funções `_item` separadas?** `load_timeseries` e `load_data` somam todos os itens da loja por dia (`store_rows[day_cols].sum()`), produzindo séries contínuas raramente iguais a zero. As funções `_item` retornam a série bruta de um único SKU — com zeros reais — e usam o preço específico do item em vez da média da loja. Isso torna a série adequada para o Croston e preserva a semântica do produto individual no XGBoost.

> **Nota:** `load_data()` retorna arrays NumPy diretamente (não DataLoaders). Para usar com PyTorch, utilize `load_data_torch()`, que aplica o mesmo pré-processamento e empacota os arrays em `DataLoader`.

### Cache de Módulo

Os três DataFrames são lidos uma única vez e mantidos em variáveis globais do módulo. Como a simulação do Flower instancia múltiplos clientes em paralelo, sem cache cada cliente releria centenas de MB em disco a cada rodada.

```python
_sales_cache    = None   # sales_train_evaluation.csv
_calendar_cache = None   # calendar.csv + datas parseadas
_prices_cache   = None   # sell_prices.csv
```

### Tratamento de NaN e Outliers

```python
# NaN: preserva continuidade temporal
daily["sales"] = daily["sales"].ffill().bfill().fillna(0.0)

# Outliers: IQR clipping em vez de remoção (mantém a série contínua)
q1, q3 = daily["sales"].quantile(0.25), daily["sales"].quantile(0.75)
iqr = q3 - q1
daily["sales"] = daily["sales"].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
```

### Feature Engineering

O pipeline constrói **12 features por timestep**:

```python
feature_cols = [
    "sin_wday", "cos_wday",               # dia da semana (period=7)
    "sin_day_of_month", "cos_day_of_month", # dia do mês (period=31)
    "sin_month", "cos_month",              # mês (period=12)
    "sin_day_of_year", "cos_day_of_year",  # dia do ano (period=365)
    "snap", "is_event",                    # programas/eventos
    "price_norm",                          # preço médio normalizado
    "sales_norm",                          # vendas normalizadas (lag)
]
```

A codificação sin/cos é necessária porque variáveis cíclicas têm uma descontinuidade artificial se passadas como inteiros: o modelo trataria "domingo (0)" e "sábado (6)" como extremos opostos quando são adjacentes no ciclo semanal.

### Normalização — Sem Data Leakage

O `StandardScaler` é ajustado **exclusivamente sobre os dados de treino**:

```python
train_end_idx = int(0.8 * n_samples) + LOOKBACK

# fit APENAS no treino — o scaler nunca vê o conjunto de teste
sales_norm[:train_end_idx] = scaler.fit_transform(sales[:train_end_idx])
sales_norm[train_end_idx:] = scaler.transform(sales[train_end_idx:])
```

Se o scaler usasse toda a série para calcular μ e σ, ele "veria" o futuro — os valores de teste influenciariam a escala dos dados de treino e inflaria artificialmente a performance dos modelos.

### Janela Deslizante — `load_data()` (XGBoost)

```python
LOOKBACK = 28

for i in range(LOOKBACK, len(data)):
    X.append(data[i - LOOKBACK : i].flatten())  # [336 features]
    y.append(targets[i])                         # venda do dia seguinte
```

Resultado: `X_train` de shape `[N, 336]`, `y_train` de shape `[N]` — ambos como `np.float32`.

### Série Bruta — `load_timeseries()` (Croston)

```python
def load_timeseries(partition_id, num_partitions=10):
    # Mesma limpeza NaN + IQR clipping que load_data()
    split = int(0.8 * len(sales))
    return sales[:split], sales[split:]  # (y_train, y_test) como np.ndarray
```

O Croston opera diretamente sobre a série 1-D, sem janela deslizante e sem normalização.

### DataLoaders PyTorch — `load_data_torch()`

```python
def load_data_torch(partition_id, num_partitions=10):
    X_train, y_train, X_test, y_test = load_data(partition_id, num_partitions)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1)),
        batch_size=32, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).unsqueeze(1)),
        batch_size=32, shuffle=False,
    )
    return train_loader, test_loader
```

---

## 4. XGBoost — Treinamento e Agregação

### Hiperparâmetros

```python
XGB_PARAMS = {
    "objective":        "reg:squarederror",  # minimiza MSE
    "eval_metric":      "rmse",
    "max_depth":        6,                   # profundidade das árvores
    "learning_rate":    0.1,
    "subsample":        0.8,                 # 80% das linhas por árvore
    "colsample_bytree": 0.8,                 # 80% das features por árvore
}
```

### Ciclo de Treino por Rodada Federada

**1. Recebe o modelo global**

O cliente desserializa o array `uint8` recebido do servidor em um `xgb.Booster`. Na rodada 1, o array está vazio — o modelo começa do zero.

**2. Treina com warm-start (5 rounds locais)**

```python
bst = xgb.train(
    XGB_PARAMS,
    dtrain,
    num_boost_round=5,
    xgb_model=xgb_model_param,  # adiciona 5 árvores ao booster recebido
)
```

O parâmetro `xgb_model` faz o warm-start: as 5 novas árvores são adicionadas sobre o ensemble recebido, sem recriar as anteriores.

**3. Avalia localmente e reporta MSE**

```python
dtest = xgb.DMatrix(self.X_test, label=self.y_test)
test_preds = bst.predict(dtest)
test_mse = float(np.mean((self.y_test - test_preds) ** 2))

return [serialized], len(self.X_train), {"mse": test_mse}
```

O MSE local é enviado junto com o booster para que o servidor possa selecionar o melhor warm-start na próxima rodada.

**4. Serializa e envia**

```python
def booster_to_ndarray(bst):
    raw = bst.save_raw(raw_format="ubj")
    return np.frombuffer(raw, dtype=np.uint8).copy()  # writeable + C-contiguous
```

O `.copy()` é necessário porque o Flower rejeita arrays não-writeable internamente.

### Agregação — `XgbEnsembleStrategy`

A estratégia combina **ensemble federado** com **warm-start pelo melhor modelo**:

#### Estrutura de Dados do Servidor

```python
class XgbEnsembleStrategy(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # cid → booster serializado (uma entrada por cliente, sem cópias antigas)
        self._client_boosters: dict[str, np.ndarray] = {}
        # cid → MSE no teste local (usado para selecionar o warm-start)
        self._client_errors: dict[str, float] = {}
```

O dicionário keyed por `cid` garante que cada cliente tenha exatamente uma entrada — quando o mesmo cliente participa em rodadas diferentes, o valor é sobrescrito em vez de acumular cópias antigas.

#### `aggregate_fit` — Acumula ensemble e escolhe warm-start

```python
def aggregate_fit(self, server_round, results, failures):
    for client_proxy, fit_res in results:
        arrays = parameters_to_ndarrays(fit_res.parameters)
        if arrays and arrays[0].size > 0:
            cid = client_proxy.cid
            self._client_boosters[cid] = arrays[0]          # substitui a cópia anterior
            self._client_errors[cid] = fit_res.metrics.get("mse", float("inf"))

    # Warm-start: o booster com menor MSE acumulado entre todos os clientes
    best_cid   = min(self._client_errors, key=lambda c: self._client_errors[c])
    warm_start = self._client_boosters[best_cid]
    others     = [self._client_boosters[cid]
                  for cid in sorted(self._client_boosters.keys())
                  if cid != best_cid]

    # parameters[0] = melhor booster (warm-start); parameters[1:] = ensemble completo
    parameters_aggregated = ndarrays_to_parameters([warm_start] + others)
    return parameters_aggregated, metrics_aggregated
```

#### `evaluate` (cliente) — Ensemble mean

```python
def evaluate(self, parameters, config):
    valid = [p for p in parameters if p.size > 0]
    boosters = [ndarray_to_booster(p) for p in valid]
    dtest = xgb.DMatrix(self.X_test, label=self.y_test)

    # Média simples entre as previsões de todos os boosters acumulados
    all_preds = np.stack([bst.predict(dtest) for bst in boosters], axis=0)
    ensemble_preds = all_preds.mean(axis=0)

    mse = float(np.mean((self.y_test - ensemble_preds) ** 2))
    mae = float(np.mean(np.abs(self.y_test - ensemble_preds)))
    return mse, len(self.X_test), {"mae": mae}
```

#### Formato dos parâmetros transmitidos

| Posição | Conteúdo | Usado em |
|---|---|---|
| `parameters[0]` | Booster com menor MSE (warm-start) | `fit()` — ponto de partida do próximo treino |
| `parameters[1:]` | Demais boosters do ensemble | `evaluate()` — membros adicionais do ensemble |

> **Por que não fazer média dos boosters?** Boosters XGBoost não podem ser somados aritmeticamente — cada árvore é uma estrutura condicional, não um vetor de pesos. A estratégia de ensemble trata cada booster como um membro independente e combina suas previsões por média simples.

---

## 5. Croston — Modelo Estatístico e Agregação

### Por que Croston?

O método de Croston foi desenvolvido para **demanda intermitente**: séries com muitos zeros intercalados por valores positivos esparsos. Modelos como XGBoost tratam os zeros como parte da tendência geral; o Croston os separa matematicamente.

### Decomposição da Série

O algoritmo suaviza dois componentes de forma independente, atualizando **apenas nos timesteps com demanda positiva**:

```
l_t = α · z_t + (1 − α) · l_{t−1}    # nível da demanda (tamanho)
p_t = β · q_t + (1 − β) · p_{t−1}    # intervalo entre demandas
```

Onde `z_t` é o tamanho da demanda não-nula e `q_t` é o intervalo em períodos desde a última ocorrência.

### Previsão — Variante SBA (implementada)

A previsão original de Croston é viesada para cima. A variante **Syntetos-Boylan (SBA)** corrige esse viés:

```
Croston Original:  F̂       = l_T / p_T
SBA (implementado): F̂_SBA  = (1 − β/2) · l_T / p_T
```

### Implementação Python

```python
for idx in non_zero_idx:
    q = float(idx - prev_pos)              # intervalo desde última demanda
    l = alpha * y[idx] + (1 - alpha) * l   # atualiza nível
    p = beta  * q     + (1 - beta)  * p    # atualiza intervalo
    prev_pos = idx

# Previsão SBA
forecast = (1 - beta / 2) * l / p
```

### Estado Federado

O estado do Croston são apenas **2 scalars**: `[demand_level, interval]`. Isso resulta em custo de comunicação mínimo — 16 bytes por cliente por rodada.

```python
def get_state(self) -> np.ndarray:
    return np.array([self.demand_level, self.interval], dtype=np.float64)

def set_state(self, state: np.ndarray) -> None:
    self.demand_level = float(state[0])
    self.interval = float(state[1])
    self.fitted = True
```

### Agregação — `CrostonFedAvgStrategy`

O servidor aplica **FedAvg sobre o estado**, usando média ponderada pelo número de amostras:

```
l_global = Σ (n_i · l_i) / Σ n_i
p_global = Σ (n_i · p_i) / Σ n_i
```

```python
def aggregate_fit(self, server_round, results, failures):
    total_samples = sum(res.num_examples for _, res in results)
    aggregated = np.zeros(2, dtype=np.float64)

    for _, fit_res in results:
        arrays = parameters_to_ndarrays(fit_res.parameters)
        if arrays and arrays[0].size == 2:
            aggregated += fit_res.num_examples * arrays[0]

    aggregated /= total_samples
    return ndarrays_to_parameters([aggregated]), {}
```

> **Limitação da abordagem atual:** a média dos parâmetros `[l, p]` entre lojas não tem interpretação estatística clara — o nível de demanda e o intervalo são estados de suavização local, específicos para a série de cada loja. Uma alternativa mais rigorosa é o **ensemble federado**: o servidor acumula um modelo por cliente e a previsão final é a média ponderada das previsões individuais (análogo ao que `XgbEnsembleStrategy` faz com os boosters).

### Warm-start Federado

| Rodada | Estado recebido | Comportamento do cliente |
|---|---|---|
| **1** | Array vazio | Cold start: `l₀ = y[first_nonzero]`, `p₀ = first_nonzero + 1` |
| **2+** | `[l_global, p_global]` | Warm-start: suavização parte do estado global agregado |

```python
if parameters and len(parameters) > 0 and parameters[0].size == 2:
    global_state = parameters[0]
    model.fit(
        self.y_train,
        init_demand_level=float(global_state[0]),
        init_interval=float(global_state[1]),
    )
else:
    model.fit(self.y_train)  # rodada 1 — cold start
```

---

## 5b. Croston Ensemble — Independência Estatística

`croston_ensemble_main.py` substitui a agregação FedAvg por um **ensemble federado sem transferência de parâmetros no treino**. A motivação é estatística: o estado `[l, p]` de um modelo Croston é um estimador de suavização local, específico para a série daquele cliente. Fazer FedAvg com outras lojas destrói essa interpretação e pode introduzir viés.

### Princípio: Separação entre Treino e Inferência

| Etapa | `croston_main.py` (FedAvg) | `croston_ensemble_main.py` (Ensemble) |
|---|---|---|
| **Treino** | Warm-start a partir do estado global médio | Cold start sempre — treino exclusivamente local |
| **Avaliação** | Um único modelo com estado global médio | Média ponderada das N previsões individuais |
| **Estado global** | `[l_global, p_global]` — média ponderada | Não existe — servidor coleta, mas não agrega |
| **Validade estatística** | Estimadores influenciados por outras lojas | Cada `[l, p]` reflete apenas a série local |

### Garantia de Independência — `configure_fit`

O servidor sobrescreve `configure_fit` para sempre enviar parâmetros vazios, independente do histórico acumulado:

```python
def configure_fit(self, server_round, parameters, client_manager):
    empty = ndarrays_to_parameters([np.array([], dtype=np.float64)])
    return super().configure_fit(server_round, empty, client_manager)
```

O cliente ignora qualquer estado recebido e sempre executa cold start:

```python
def fit(self, parameters, config):
    model = CrostonForecaster(alpha=CROSTON_ALPHA, beta=CROSTON_BETA, variant=CROSTON_VARIANT)
    model.fit(self.y_train)   # cold start sempre — parâmetros do servidor são ignorados
    _, mae = model.evaluate(self.y_test)
    return [model.get_state()], len(self.y_train), {"mae": mae}
```

### Coleta dos Estados — `CrostonEnsembleStrategy`

O servidor acumula um estado por `cid` (sem fazer média):

```python
for client_proxy, fit_res in results:
    arrays = parameters_to_ndarrays(fit_res.parameters)
    if arrays and arrays[0].size == 2:
        cid = client_proxy.cid
        self._client_states[cid] = arrays[0]           # substitui — não agrega
        self._client_errors[cid] = fit_res.metrics.get("mae", float("inf"))
```

### Pesos do Ensemble — Inverso do MAE

```
w_i = (1 / (MAE_i + ε)) / Σ_j (1 / (MAE_j + ε))
```

Modelos com menor MAE local recebem mais peso na previsão global:

```python
inv_errors = 1.0 / (errors + 1e-9)
weights = (inv_errors / inv_errors.sum()).astype(np.float64)
```

### Transmissão para Avaliação

```
parameters[0]   = pesos do ensemble   (float64, tamanho N)
parameters[1:]  = N estados [l, p]    (um por cliente, sorted by cid)
```

### Previsão do Ensemble (cliente)

```python
predictions = [model_i.predict() for model_i in ensemble_models]
ensemble_pred = np.average(predictions, weights=weights)
```

### Quando Usar Cada Abordagem

| Cenário | Recomendação |
|---|---|
| Lojas com padrões de demanda similares | `croston_main.py` (FedAvg pode se beneficiar do estado compartilhado) |
| Lojas com padrões heterogêneos (ex: CA vs. WI) | `croston_ensemble_main.py` (cada modelo captura seu padrão local) |
| Foco em validade estatística dos estimadores | `croston_ensemble_main.py` (treino sempre independente) |

---

## 6. Baselines Centralizados

`centralized.py` implementa versões centralizadas de ambos os modelos. Serve como **teto de referência**: o melhor resultado teoricamente alcançável por cada abordagem de modelagem, sem restrições de privacidade ou federação.

Ambas as funções usam as APIs sem partição (`load_all_data`, `load_all_timeseries`) e reportam métricas em **unidades brutas de vendas**, tornando XGBoost e Croston diretamente comparáveis.

```bash
# Modo padrão — todas as 10 lojas
python centralized.py

# Modo produto único — XGBoost + Croston num único SKU
python centralized.py --item FOODS_3_090 --store CA_1
```

No modo produto único, `centralized.py` chama `run_single_product(item_id, store_id)`, que usa `load_data_item` e `load_timeseries_item` para carregar a série bruta do item. O resultado é impresso em uma tabela comparativa idêntica à dos scripts federados, permitindo comparar diretamente com os experimentos federados no mesmo produto.

### Comparabilidade das Métricas

| Modelo | Escala original das previsões | Como as métricas são calculadas |
|---|---|---|
| **XGBoost** | Normalizada (StandardScaler por loja) | `inverse_transform` por scaler de loja → escala bruta |
| **Croston** | Bruta (série sem normalização) | Direto — já está em escala bruta |

Sem o `inverse_transform`, o MSE do XGBoost seria algo como `0.42` (unitless) enquanto o Croston reportaria `18 000` (unidades de vendas²) — impossível comparar. Após a transformação inversa, ambos estão na mesma escala.

### XGBoost Centralizado — `run_centralized_xgboost()`

Usa `load_all_data()` para carregar todas as lojas de uma vez. Um único modelo é treinado no conjunto concatenado com 50 rounds, aprende padrões cross-loja e reporta métricas em escala bruta via `inverse_transform`:

```python
X_all_train, y_all_train, X_all_test, y_all_test, \
    train_sizes, test_sizes, sales_scalers = load_all_data()

bst = xgb.train(XGB_PARAMS, dtrain, num_boost_round=50)
preds_norm = bst.predict(dtest)

# Inverse_transform por loja → escala bruta de vendas
for i, (n, scaler) in enumerate(zip(test_sizes, sales_scalers)):
    store_pred_raw = scaler.inverse_transform(preds_norm[offset:offset+n].reshape(-1,1))
    store_actual_raw = scaler.inverse_transform(y_all_test[offset:offset+n].reshape(-1,1))
    # MSE/MAE em unidades brutas
```

**Por que é o teto?** O modelo centralizado tem acesso simultâneo a todas as lojas e usa 50 rounds (vs. 5 locais no federado), capturando padrões que nenhum cliente federado individual consegue observar.

### Croston Centralizado — `run_centralized_croston()`

Usa `load_all_timeseries()` para carregar todas as séries de uma vez. Cada loja tem seu próprio modelo Croston. O ensemble é ponderado por `1/MAE` (mesmo critério do `croston_ensemble_main.py`):

```python
y_trains, y_tests = load_all_timeseries()

for y_train, y_test in zip(y_trains, y_tests):
    m = CrostonForecaster(alpha=alpha, beta=beta, variant=variant)
    m.fit(y_train)
    _, local_mae = m.evaluate(y_test)
    inv_mae_weights.append(1.0 / (local_mae + 1e-9))

# Ensemble ponderado por 1/MAE — igual ao cenário federado
weights = [w / sum(inv_mae_weights) for w in inv_mae_weights]
ensemble_forecast = sum(w * m.predict() for w, m in zip(weights, models))
```

**Relação com o federado:** após rodadas suficientes para todos os clientes participarem ao menos uma vez, o `croston_ensemble_main.py` converge para este resultado. A diferença na prática vem de `fraction_fit < 1.0` — nem todos os clientes participam em toda rodada.

---

## 7. Validade do Pipeline de Avaliação

### Problemas identificados e correções aplicadas

#### 1. IQR calculado sobre a série completa (leakage) — CORRIGIDO

**Problema:** `load_timeseries()` e `load_data()` calculavam os quantis (Q1, Q3) sobre a série inteira, incluindo o período de teste. Os limites de clipping vazavam informação futura para os dados de treino.

**Correção:** IQR agora é calculado exclusivamente sobre os primeiros 80% (treino) e aplicado à série completa:
```python
# ANTES — leakage
q1, q3 = daily["sales"].quantile(0.25), daily["sales"].quantile(0.75)

# DEPOIS — sem leakage
train_sales_raw = daily["sales"].iloc[:train_end_raw]
q1, q3 = train_sales_raw.quantile(0.25), train_sales_raw.quantile(0.75)
```

#### 2. Conjunto de teste usado em decisões de treino (leakage) — CORRIGIDO

**Problema:** Tanto XGBoost quanto Croston ensemble reportavam métricas do **conjunto de teste** dentro de `fit()`, que o servidor usava para selecionar warm-start e calcular pesos do ensemble. O teste deve ser invisível durante o processo de treinamento.

**XGBoost** (`main.py`): o MSE do teste selecionava o booster de warm-start.
**Croston ensemble** (`croston_ensemble_main.py`): o MAE do teste ponderava o ensemble.
**Croston centralizado** (`centralized.py`): o MAE do teste era usado para os pesos.

**Correção:** todas essas métricas passam a usar o **conjunto de treino**. O `y_test` é usado exclusivamente em `evaluate()`.

#### 3. Croston aplicado a demanda não-intermitente — LIMITAÇÃO NO MODO LOJA

**Problema:** O método de Croston foi desenvolvido para séries com muitos zeros (demanda intermitente). No modo padrão (loja inteira), as vendas são **somadas de todos os itens da loja** por dia:
```python
sales = store_rows[day_cols].sum().values  # centenas de itens somados
```
A soma de centenas de itens raramente é zero — a série é **contínua e sazonal**, não intermitente. O Croston não captura sazonalidade semanal nem tendência, resultando em uma previsão constante para o horizonte inteiro.

**Impacto no modo loja:** o Croston produz resultados inferiores ao XGBoost não por ser um modelo federado inferior, mas por ser o modelo errado para o dado agregado.

**Modo produto único (`ITEM_ID` / `--item`):** ao treinar sobre a série bruta de um único SKU (`load_timeseries_item`), a série apresenta zeros reais — o Croston opera no regime para o qual foi projetado. Nesse modo a comparação com XGBoost é válida.

**Alternativas ao Croston para demanda agregada de loja:**
- Simple Exponential Smoothing (SES) — sem sazonalidade
- Holt-Winters — com tendência e sazonalidade
- SARIMA — sazonal com componente autorregressivo

#### 4. Tarefa de previsão diferente entre modelos

| Aspecto | XGBoost | Croston |
|---|---|---|
| Tipo de previsão | Dinâmica (1 valor por dia) | Estática (constante para todo o horizonte) |
| Número de previsões | `T_test` (um por timestep) | 1 (aplicada a todos os `T_test` dias) |
| Captura sazonalidade | Sim (via features cíclicas) | Não (modelo estacionário) |
| Captura tendência | Sim (via janela deslizante) | Não |

Mesmo com métricas na mesma escala (bruta de vendas), o XGBoost resolve um problema estritamente mais difícil — e ainda assim obtém melhor resultado, evidenciando a inadequação do Croston para demanda agregada.

---

## 8. Guia de Interpretação das Métricas

Todas as métricas estão em **escala bruta de vendas** (unidades/dia), exceto R² que é adimensional.

### O que cada métrica mede

| Métrica | Fórmula | Unidade | Interpreta-se como… |
|---|---|---|---|
| **RMSE** | `√(Σ(y−ŷ)² / n)` | vendas/dia | Erro típico de previsão. Se RMSE=150, o modelo erra em média ~150 unidades por dia |
| **MAE** | `Σ|y−ŷ| / n` | vendas/dia | Erro médio absoluto. Mais robusto a dias atípicos (picos de promoção, etc.) |
| **MSE** | `Σ(y−ŷ)² / n` | (vendas/dia)² | Base interna de otimização. Difícil de ler diretamente — prefira RMSE |
| **R²** | `1 − SS_res/SS_tot` | adimensional [−∞, 1] | Fração da variação de demanda explicada pelo modelo |

### Como ler o R²

```
R² = 1.0   → previsão perfeita
R² = 0.0   → o modelo não é melhor do que prever sempre a média histórica
R² < 0.0   → o modelo é pior do que prever a média (comum no Croston sobre dados sazonais)
```

O R² é calculado por loja e globalmente. No cenário federado, o valor reportado é a **média ponderada** por número de amostras entre os clientes avaliados naquela rodada — uma aproximação do R² global real.

### O que significa "Loss" no histórico do Flower

```python
hist.losses_distributed  # ex: [(1, 18432.5), (2, 17210.3), ...]
```

Cada tupla é `(rodada, MSE_ponderado)`. É o mesmo MSE reportado como `loss` pelo cliente em `evaluate()`. O valor está em (vendas/dia)² — tire a raiz quadrada para obter o RMSE em escala legível.

### Exemplo de leitura prática

```
XGBoost  → RMSE=120.4  MAE=89.2  R²=0.61
Croston  → RMSE=310.7  MAE=248.3 R²=-0.42
```

- **XGBoost**: erra ~120 unidades/dia em média; explica 61% da variação de demanda.
- **Croston**: erra ~311 unidades/dia; R² negativo indica que prever a média histórica seria melhor. Isso reflete a inadequação do método para demanda agregada contínua e sazonal (ver seção 7).

### RMSE vs MAE — quando divergem

Se `RMSE >> MAE`, existem dias com erros muito grandes (ex: picos de promoção). Se forem próximos, os erros são distribuídos de forma uniforme.

---

## 9. XGBoost vs Croston — Comparação

| Dimensão | XGBoost (`main.py`) | Croston FedAvg (`croston_main.py`) | Croston Ensemble (`croston_ensemble_main.py`) |
|---|---|---|---|
| Tipo de modelo | Gradient boosted trees | Exponential smoothing | Exponential smoothing |
| Input de treino | Janela 28 dias × 12 features (336 dims) | Série 1-D bruta | Série 1-D bruta |
| Estado federado | Booster serializado (~centenas de KB) | 2 floats: `[l, p]` (16 bytes) | 2 floats: `[l, p]` por cliente |
| Agregação servidor | Ensemble — acumula 1 booster por cid | FedAvg — média ponderada do estado | Coleta por cid — sem média |
| Warm-start | Booster com menor MSE (acumula árvores) | Estado global médio `[l_global, p_global]` | Nenhum — cold start sempre |
| Influência cruzada entre lojas | Sim (via warm-start) | Sim (estado médio guia o treino) | Não (treino estritamente local) |
| Custo de comunicação | Alto (booster completo) | Mínimo (16 bytes) | Mínimo (16 bytes × N clientes) |
| Melhor para | Demanda contínua com padrões complexos | Lojas com padrões similares | Lojas com padrões heterogêneos |
| Previsão output | Valor normalizado por timestep | Escalar constante | Escalar ponderado do ensemble |
| Baseline centralizado | `run_centralized_xgboost()` | `run_centralized_croston()` | `run_centralized_croston()` |

---

## 9. Como Executar

### Pré-requisitos

```
data/
├── sales_train_evaluation.csv
├── calendar.csv
└── sell_prices.csv
```

### Instalar Dependências

```bash
pip install -r requirements.txt
```

### Baselines Centralizados (referência)

```bash
# Todas as 10 lojas (modo padrão)
python centralized.py

# Produto único — XGBoost + Croston num único SKU
python centralized.py --item FOODS_3_090 --store CA_1
```

Executa `run_centralized_xgboost()` e `run_centralized_croston()` em sequência (modo padrão), ou `run_single_product()` (modo `--item`), e imprime MSE/MAE global e por loja para cada modelo.

### Experimento XGBoost Federado

```bash
# Modo padrão — demanda agregada por loja
python main.py

# Produto único — defina ITEM_ID no topo de main.py antes de executar
# ITEM_ID = "FOODS_3_090"   ← altere aqui
python main.py
```

> **Nota — dependência Ray no Windows:** a simulação do Flower usa Ray como backend. Ray é experimental no Windows e pode não estar disponível. Se ocorrer o erro `KeyError: 'ray'`, o problema está no `run_simulation` do Flower — uma versão sem Ray (loop de simulação manual em `simulation.py`) está planejada.

### Experimento Croston Federado (FedAvg)

```bash
# Modo padrão — demanda agregada por loja
python croston_main.py

# Produto único — defina ITEM_ID no topo de croston_main.py antes de executar
# ITEM_ID = "FOODS_3_090"   ← altere aqui
python croston_main.py
```

### Experimento Croston Ensemble Federado (sem agregação de parâmetros)

```bash
# Modo padrão — demanda agregada por loja
python croston_ensemble_main.py

# Produto único — defina ITEM_ID no topo de croston_ensemble_main.py antes de executar
# ITEM_ID = "FOODS_3_090"   ← altere aqui
python croston_ensemble_main.py
```

> **Modo produto único nos scripts federados:** quando `ITEM_ID` está definido, cada um dos 10 clientes carrega a série bruta daquele SKU na **sua própria loja** (`FOODS_3_090 @ CA_1`, `FOODS_3_090 @ CA_2`, …). A série tem zeros reais, tornando o Croston estatisticamente adequado. Os scripts usam `load_timeseries_item(ITEM_ID, STORE_IDS[partition_id])` ou `load_data_item(ITEM_ID, STORE_IDS[partition_id])` conforme o modelo.

### Saída Esperada por Rodada (XGBoost)

```
[Rodada  1] Ensemble:  3 boosters | warm-start=cid node:0 (MSE=0.423817)
[Rodada  2] Ensemble:  5 boosters | warm-start=cid node:2 (MSE=0.391042)
...
[Rodada 10] Ensemble: 10 boosters | warm-start=cid node:7 (MSE=0.318654)

=== Histórico de Treinamento ===
Losses distribuídas: [(1, 0.42), (2, 0.39), ...]
Métricas distribuídas: [(1, {'mae': 0.51, 'mse': 0.42}), ...]
```

### Saída Esperada por Rodada (Croston FedAvg)

```
[Rodada  1] Estado global agregado → demand_level=142.3021, interval=3.8740
[Rodada  2] Estado global agregado → demand_level=139.7654, interval=3.9102
...
[Rodada 10] Estado global agregado → demand_level=137.1203, interval=4.0551

=== Histórico de Treinamento (Croston Federado) ===
Losses distribuídas (MSE): [(1, 18432.5), (2, 17821.3), ...]
Métricas distribuídas (MAE): [(1, {'mae': 82.4}), ...]
```

### Saída Esperada por Rodada (Croston Ensemble)

```
[Rodada  1] Ensemble:  3 modelos | melhor=cid node:2 (MAE=71.3412)
[Rodada  2] Ensemble:  5 modelos | melhor=cid node:2 (MAE=71.3412)
...
[Rodada 10] Ensemble: 10 modelos | melhor=cid node:7 (MAE=68.9041)

=== Histórico de Treinamento (Croston Ensemble Federado) ===
Losses distribuídas (MSE): [(1, 17210.3), (2, 16854.1), ...]
Métricas distribuídas (MAE): [(1, {'mae': 75.2}), ...]
```

### Configurações Principais

| Parâmetro | Arquivo | Padrão | Descrição |
|---|---|---|---|
| `NUM_SERVER_ROUNDS` | `main.py` / `croston_main.py` | `10` | Rodadas federadas |
| `fraction_fit` | servidor | `0.3` | Fração de clientes por rodada de treino |
| `LOOKBACK` | `dataset.py` | `28` | Janela deslizante em dias (XGBoost) |
| `CROSTON_ALPHA` | `croston_main.py` / `croston_ensemble_main.py` | `0.1` | Suavização do nível de demanda |
| `CROSTON_BETA` | `croston_main.py` / `croston_ensemble_main.py` | `0.1` | Suavização do intervalo entre demandas |
| `CROSTON_VARIANT` | `croston_main.py` / `croston_ensemble_main.py` | `"sba"` | `"sba"` (recomendado) ou `"original"` |
| `num_boost_round` | `centralized.py` | `50` | Rounds do XGBoost centralizado |
| `ITEM_ID` | `main.py` / `croston_main.py` / `croston_ensemble_main.py` | `None` | Se definido, cada cliente usa esse SKU na sua loja em vez da soma da loja inteira |
