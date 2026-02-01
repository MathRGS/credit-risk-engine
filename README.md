# 🛡️ Credit Risk Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production-success?style=for-the-badge)

> Uma solução analítica *full-stack* para **Modelagem de Risco de Crédito**, **Gestão de Portfólio** e **Stress Testing Macroeconômico**.

---

## 🎯 Sumário Executivo

Este projeto preenche a lacuna entre a **Economia Financeira** e a **Engenharia de Software**. Trata-se de uma aplicação web totalmente conteinerizada, projetada para simular, analisar e estressar carteiras de crédito utilizando modelos estocásticos e calibração estatística.

Diferente de dashboards convencionais, esta aplicação implementa um **Motor de Risco Proprietário (Risk Engine)** baseado em Regressão Logística para estimativa de PD (*Probability of Default*) e métodos de Monte Carlo para geração de cenários de portfólio.

### 🔗 Live Demo
**https://risk.matheusrocha.cloud**

---

## 📊 Funcionalidades Principais

### 1. Simulação Estocástica de Portfólio
- **Motor de Monte Carlo:** Gera carteiras sintéticas (N = 2000+) com distribuições realistas para Renda, Dívida e Score de Bureau.  
- **Injeção de Risco de Cauda (Fat-Tail):** Simula eventos extremos injetando outliers de alto risco (Rating D) para testar a resiliência do modelo.

### 2. Métricas de Risco Avançadas (Framework de Basileia)
Cálculo em tempo real das principais métricas bancárias:
- **PD (Probability of Default)** — Calibrada via Regressão Logística  
- **LGD (Loss Given Default)** — Calibração dinâmica via controles de UI  
- **EAD (Exposure at Default)** — Volume total exposto ao risco  
- **EL (Expected Loss)** → `EL = PD × LGD × EAD`

### 3. Stress Testing Macroeconômico
- Simulação de cenários econômicos adversos (Recessão Leve, Crise Severa)  
- Aplicação de choques dinâmicos nas PDs (+20%, +50%)  
- Recálculo automático da perda esperada e impacto financeiro  

### 4. Simulador de Underwriting (Concessão)
- Avaliação de crédito individual em tempo real  
- Classificação automática de Rating (A a D)  
- Visualização de risco com gráficos Gauge (velocímetro)

### 5. UX/UI Mobile-First
- Interface moderna com customização visual em CSS  
- Controles de risco otimizados para usabilidade em dispositivos móveis  

---

## 🛠️ Stack Tecnológico e Arquitetura

Arquitetura preparada para evolução em microsserviços utilizando Docker.

| Camada | Tecnologia | Função |
|--------|------------|--------|
| **Core Engine** | Python, Scikit-Learn | Modelagem estatística de risco |
| **Frontend** | Streamlit | Interface web reativa |
| **Visualização** | Plotly | Gráficos financeiros interativos |
| **Processamento** | Pandas, NumPy | Simulação e manipulação de dados |
| **Infraestrutura** | Docker Compose | Containerização e deploy |

---

## 🧮 Modelo Matemático

A lógica central baseia-se na fórmula de **Perda Esperada (Expected Loss)**:

\[
EL = \sum_{i=1}^{n} (PD_i \times LGD \times EAD_i)
\]

Onde:

- **PDᵢ** = Probabilidade de Default do cliente *i*  
\[
P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 Score + \beta_2 Renda + ...)}}
\]

- **LGD** = Loss Given Default (padrão regulatório: 45%)  
- **EADᵢ** = Exposure at Default (valor do empréstimo)

---

## 🚀 Instalação e Execução

### ✅ Pré-requisitos
- Docker  
- Docker Compose  
- Git  

---

### 📥 1. Clonar o repositório

```bash
git clone https://github.com/MathRGS/credit-risk-engine.git
cd credit-risk-engine
```

---

### 🐳 2. Rodar com Docker Compose

Este comando constrói a imagem e inicia a aplicação na porta **8501**:

```bash
docker compose up -d --build
```

---

### 🌐 3. Acessar

Abra no navegador:

```
http://localhost:8501
```

---

## 📂 Estrutura do Projeto

```plaintext
├── app.py                 # Aplicação principal (Streamlit UI)
├── risk_engine.py         # Motor de risco (modelagem e simulação)
├── credit_model.pkl       # Modelo de Machine Learning serializado
├── requirements.txt       # Dependências Python
├── Dockerfile             # Configuração de container
├── docker-compose.yml     # Orquestração dos serviços
├── deploy.sh              # Script de deploy
└── README.md              # Documentação
```

---

## 👨‍💻 Autor

**Matheus Rocha**  
*Economista | Especialista em Tesouraria | Fullstack Developer*

Unindo finanças quantitativas e engenharia de software para construir soluções **fintech escaláveis**.

🔗 Conecte-se:
- LinkedIn: https://www.linkedin.com/in/matheus-rocha-4a616320a/  
- Portfólio: https://matheusrocha.cloud  

---

© 2026 Credit Risk Intelligence Platform. Todos os direitos reservados.
