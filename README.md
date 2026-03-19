#  Customising Gymnasium Environments and Reinforcement Learning with Stable-Baselines3

##  Visão Geral do Projeto

Este projeto explora, implementa e analisa agentes de **Aprendizagem por Reforço** (*Reinforcement Learning — RL*) treinados num **ambiente Gymnasium customizado**, com o objetivo de compreender como o **design do ambiente** e o **reward shaping** afetam a estabilidade da aprendizagem e o desempenho final do agente.

Utilizando o ambiente **LunarLander-v3** como *baseline*, foram introduzidas modificações controladas ao nível da **função de recompensa**, e os agentes foram treinados com recurso à biblioteca **Stable-Baselines3 (SB3)**. O projeto segue uma metodologia experimental rigorosa, incluindo **benchmark de algoritmos**, **tuning de hiperparâmetros**, **validação multi-seed** e **comparação estatística** entre o ambiente original e o ambiente customizado.



##  Questão de Investigação

> **De que forma alterações na função de recompensa influenciam a estabilidade, a convergência e o desempenho final de um agente de Aprendizagem por Reforço?**


##  Estrutura do Projeto 

```text
TRABALHOSISFINAL/
├── final.ipynb
│   └── Main notebook containing:
│       ├── Environment analysis and customisation
│       ├── Algorithm benchmark (A2C, PPO, DQN)
│       ├── PPO hyperparameter tuning
│       ├── Training of PPO variants
│       ├── Multi-seed validation
│       └── Statistical analysis and conclusions
│
├── final_comparison/
│   ├── original/
│   │   └── seed_*/
│   │       ├── monitor.csv
│   │       └── final_model.zip
│   └── customizado/
│       └── seed_*/
│           ├── monitor.csv
│           └── final_model.zip
│   └── Multi-seed validation logs and final models
│
├── logs_A2C/
│   └── monitor.csv
│   └── A2C benchmark training logs
│
├── logs_DQN/
│   └── monitor.csv
│   └── DQN benchmark training logs
│
├── logs_PPO/
│   └── monitor.csv
│   └── PPO benchmark training logs
│
├── ppo_variants/
│   ├── PPO_Stable/
│   │   └── monitor.csv
│   │   └── Stable PPO variant (final model)
│   ├── PPO_Aggressive/
│   │   └── monitor.csv
│   │   └── Aggressive PPO variant
│   └── PPO_Optimized/
│       └── monitor.csv
│       └── Optimized PPO variant (individual optimum)
│
├── tuning_lr/
│   └── PPO_lr_*/
│       └── monitor.csv
│   └── Learning rate tuning experiments
│
├── tuning_nsteps/
│   └── PPO_nsteps_*/
│       └── monitor.csv
│   └── n_steps tuning experiments
│
├── tuning_batch/
│   └── PPO_bs_*/
│       └── monitor.csv
│   └── Batch size tuning experiments
│
├── tuning_gamma/
│   └── PPO_gamma_*/
│       └── monitor.csv
│   └── Discount factor (γ) tuning experiments
│
├── tuning_lambda/
│   └── PPO_lambda_*/
│       └── monitor.csv
│   └── GAE lambda (λ) tuning experiments
│
├── tuning_clip/
│   └── PPO_clip_*/
│       └── monitor.csv
│   └── Clip range tuning experiments
│
├── tuning_ent/
│   └── PPO_ent_*/
│       └── monitor.csv
│   └── Entropy coefficient (ent_coef) tuning experiments
│
├── tuning_vf/
│   └── PPO_vf_*/
│       └── monitor.csv
│   └── Value function coefficient (vf_coef) tuning experiments
│
├── tuning_grad/
│   └── PPO_grad_*/
│       └── monitor.csv
│   └── Gradient clipping (max_grad_norm) tuning experiments
│
├── comparacao_final_variantes_ppo.png
│   └── Comparison plot of PPO variants
│
├── comparacao_final_stress_deriva.gif
│   └── Side-by-side policy visualisation with stress test
│
├── comparacao_final_treinados.gif
│   └── Trained policies visual comparison
│
├── comparacao_final_treinados_stress.gif
│   └── Trained policies with stress test
│
├── random_agent_demo.gif
│   └── Random agent demonstration
│
├── tensorBoard.png
│   └── TensorBoard training curves
│
├── Apresentação.pptx
│   └── Final project presentation slides
│
├── requirements.txt
│   └── Python dependencies
│
└── README.md
    └── Project documentation

##  Descrição do Ambiente

### Ambiente Original — `LunarLander-v3`

- **Espaço de observações:** 8 variáveis contínuas  
- **Espaço de ações:** Discreto (4 ações)  
- **Objetivo:** Aterrar a nave de forma segura numa plataforma designada  
- **Função de recompensa (*reward shaping*):** Incentiva estabilidade, precisão e eficiência no uso de combustível  



### Ambiente Customizado — `LunarLanderCustom-v0`

O ambiente customizado modifica **exclusivamente a função de recompensa**, mantendo inalterados o espaço de observações e o espaço de ações.

#### Penalizações adicionadas:
- Ângulo absoluto elevado  
- Velocidade angular elevada  
- Velocidade horizontal elevada  
- Uso excessivo dos *thrusters* laterais  

Estas penalizações incentivam trajetórias mais suaves e aterragens mais estáveis, aumentando simultaneamente a dificuldade da tarefa.



##  Algoritmos de Reinforcement Learning

### Benchmark de Algoritmos

Foram inicialmente testados três algoritmos da biblioteca **Stable-Baselines3 (SB3)** no ambiente original:

- **A2C** — *baseline* do tipo Actor-Critic  
- **PPO** — otimização de políticas estável  
- **DQN** — *Deep Q-Learning* (ações discretas)  

**Resultado:** o algoritmo **PPO** apresentou o melhor desempenho e foi selecionado para as fases seguintes do projeto.



##  Tuning de Hiperparâmetros (PPO)

O *tuning* dos hiperparâmetros foi realizado de forma incremental, ajustando **um parâmetro de cada vez**.

### Hiperparâmetros ajustados:
- *Learning rate*  
- `n_steps`  
- `batch_size`  
- Fator de desconto (γ)  
- *GAE lambda* (λ)  
- `clip_range`  
- Coeficiente de entropia (`ent_coef`)  
- Coeficiente da função de valor (`vf_coef`)  
- `max_grad_norm`  

Cada configuração foi treinada durante **100 000 timesteps**.



##  Variantes do PPO (1M Timesteps)

Foram treinadas e analisadas três variantes completas do PPO:

### PPO Estável (Modelo Final)
- Atualizações conservadoras  
- Elevada estabilidade  
- Única variante a convergir de forma consistente  

### PPO Agressivo
- Atualizações mais rápidas  
- Aprendizagem instável  

### PPO Otimizado (Ótimo Individual)
- Combinação dos melhores valores individuais  
- Falhou devido a interações negativas entre hiperparâmetros  



##  Validação Multi-Seed

O modelo final **PPO Estável** foi treinado com **5 *seeds* diferentes** em:

- **Ambiente original** (baseline)  
- **Ambiente customizado** (penalizado)  

Cada treino utilizou **2 milhões de timesteps**.



##  Resumo dos Resultados

| Ambiente      | Recompensa Média (Últimos 100 Episódios) | Desvio Padrão |
|---------------|------------------------------------------|---------------|
| Original      | +209.54                                  | 102.51        |
| Customizado  | +35.80                                   | 138.59        |



##  Análise Qualitativa

Foram geradas visualizações das políticas aprendidas e *stress tests* para apoiar a interpretação dos resultados.

>  Estas visualizações são ilustrativas e não substituem a análise quantitativa.



##  Principais Conclusões

- O *reward shaping* tem um impacto significativo no comportamento do agente  
- Penalizações excessivas podem prejudicar a exploração e a convergência  
- O design do ambiente é tão crítico quanto a escolha do algoritmo  
- A validação multi-seed é essencial para uma avaliação robusta em RL  



##  Como Executar o Projeto

1. **Instalar dependências:**
   ```bash
   pip install -r requirements.txt
2. Abrir o notebook:
jupyter notebook final.ipynb

3.Executar o notebook do início ao fim para reproduzir todas as experiências.
 O treino pode demorar várias horas em CPU.

Tecnologias Utilizadas:
- Python 3
- Gymnasium
- Stable-Baselines3
- NumPy, Pandas
- Matplotlib
- TensorBoard