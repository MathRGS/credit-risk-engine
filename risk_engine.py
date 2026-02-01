import pandas as pd
import numpy as np
import logging
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV

# ---------------------------------------------------
# LOGGING PROFISSIONAL
# ---------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CreditRiskEngine")


class CreditRiskEngine:
    def __init__(self, model_path="credit_model.pkl", lgd=0.45):
        self.model_path = model_path
        self.lgd = lgd
        self.features = ['renda_mensal', 'score_serasa', 'idade', 'divida_total']
        self.scaler = StandardScaler()
        self.model = None
        self.gini = None
        self.ks = None

    # ---------------------------------------------------
    # DADOS SINTÉTICOS REALISTAS
    # ---------------------------------------------------
    def _generate_synthetic_data(self, n_samples=8000):
        rng = np.random.default_rng()

        # AJUSTE 1: Média do Score subiu de 620 para 680 (População melhor)
        # AJUSTE 2: Desvio padrão maior (150) para ter tanto gente muito boa quanto muito ruim
        score = rng.normal(680, 150, n_samples).clip(0, 1000)
        
        # Renda correlacionada levemente com o score para realismo
        renda_base = rng.lognormal(mean=8.4, sigma=0.55, size=n_samples)
        renda = renda_base * (1 + (score - 600)/1000) # Quem tem score alto ganha um pouco mais
        
        idade = rng.integers(18, 80, n_samples)
        divida = rng.exponential(6000, n_samples)

        # Cálculo do gabarito (Inadimplente real)
        risco_base = (
            (score < 450).astype(int) * 0.40 +          # Score baixo pesa mais
            (divida / renda > 1.0).astype(int) * 0.25 + # Endividado
            (idade < 23).astype(int) * 0.10 +           # Muito jovem
            rng.normal(0, 0.1, n_samples)               # Ruído aleatório
        )

        # Sigmoid para definir probabilidade real de default
        prob_default = 1 / (1 + np.exp(-(-2.5 + risco_base * 3.5)))
        inadimplente = rng.binomial(1, prob_default)

        return pd.DataFrame({
            'renda_mensal': renda,
            'score_serasa': score,
            'idade': idade,
            'divida_total': divida,
            'inadimplente': inadimplente
        })



    # ---------------------------------------------------
    # MÉTRICAS DE VALIDAÇÃO (FINTECH PADRÃO)
    # ---------------------------------------------------
    def _calculate_ks(self, y_true, y_score):
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        ks = max(tpr - fpr)
        return ks

    # ---------------------------------------------------
    # TREINAMENTO
    # ---------------------------------------------------
    def train_model(self, data_path=None):

        if data_path and os.path.exists(data_path):
            df = pd.read_csv(data_path)
            logger.info("Dados reais carregados.")
        else:
            logger.warning("Base real não encontrada. Usando dados sintéticos.")
            df = self._generate_synthetic_data()

        X = df[self.features]
        y = df['inadimplente']

        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )

        base_model = LogisticRegression(class_weight='balanced', solver='liblinear')
        calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
        calibrated_model.fit(X_train, y_train)

        probs = calibrated_model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, probs)
        gini = 2 * auc - 1
        ks = self._calculate_ks(y_test, probs)

        self.model = calibrated_model
        self.gini = gini
        self.ks = ks

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'gini': gini,
            'ks': ks
        }, self.model_path)

        logger.info(f"Modelo treinado | Gini: {gini:.2%} | KS: {ks:.2%}")

        return gini, ks

    # ---------------------------------------------------
    # LOAD
    # ---------------------------------------------------
    def load_model(self):
        if not os.path.exists(self.model_path):
            logger.info("Modelo não encontrado. Treinando novo modelo...")
            self.train_model()

        artifacts = joblib.load(self.model_path)
        self.model = artifacts['model']
        self.scaler = artifacts['scaler']
        self.gini = artifacts.get('gini', 0.5)
        self.ks = artifacts.get('ks', 0.3)

        logger.info(f"Modelo carregado | Gini: {self.gini:.2%} | KS: {self.ks:.2%}")
        return self.gini, self.ks

    # ---------------------------------------------------
    # PREDIÇÃO
    # ---------------------------------------------------
    def predict(self, input_data: pd.DataFrame):
        if self.model is None:
            self.load_model()

        for col in self.features:
            if col not in input_data.columns:
                input_data[col] = 0

        X_scaled = self.scaler.transform(input_data[self.features])
        return self.model.predict_proba(X_scaled)[:, 1]

    # ---------------------------------------------------
    # SIMULAÇÃO DE CARTEIRA
    # ---------------------------------------------------
    def simulate_portfolio(self, n_contracts=2000, stress_factor=1.0):
        if self.model is None:
            self.load_model()

        df = self._generate_synthetic_data(n_contracts)

        # AJUSTE 3: Quem tem score alto pede empréstimos MAIORES (Crédito Prime)
        # Isso garante que o RATING A tenha "Volume Financeiro" alto no gráfico
        base_loan = np.random.lognormal(9, 0.8, n_contracts)
        score_multiplier = (df['score_serasa'] / 500).clip(0.5, 3.0) # Score 800 pega 1.6x mais dinheiro que Score 500
        df['valor_solicitado'] = base_loan * score_multiplier

        # Predição do modelo
        X_scaled = self.scaler.transform(df[self.features])
        df['pd'] = self.model.predict_proba(X_scaled)[:, 1]

        # AJUSTE 4: Baixei a média alvo de 10% para 5.5% (Padrão Banco Varejo Saudável)
        # Isso impede que o fator de calibração "estrague" os clientes bons
        target_pd_mean = 0.055 
        current_mean = df['pd'].mean()
        
        # Fator de calibração suave
        scaling_factor = target_pd_mean / current_mean
        
        # Aplica Stress e Calibração
        # Aplica Stress e Calibração
        df['pd'] = df['pd'] * scaling_factor * stress_factor
        
        # --- A CORREÇÃO AQUI (FAT TALI RISK) ---
        # Força 5% da carteira a ser "D (Crítico)" artificialmente
        # Isso simula fraudes ou quebras súbitas que o modelo normal não pega
        n_bad_apples = int(n_contracts * 0.05) 
        idx_bad = np.random.choice(df.index, n_bad_apples, replace=False)
        # Define PD desses caras entre 20% e 50%
        df.loc[idx_bad, 'pd'] = np.random.uniform(0.20, 0.50, n_bad_apples)
        # ---------------------------------------

        # Trava para não quebrar a matemática
        df['pd'] = df['pd'].clip(0.001, 0.999)

        df['loss_given_default'] = self.lgd
        df['expected_loss'] = df['pd'] * df['loss_given_default'] * df['valor_solicitado']

        # Classificação de Rating (Mantive igual)
        def classificar_rating(pd):
            if pd < 0.03: 
                return 'A (Baixo Risco)'
            elif pd < 0.08: 
                return 'B (Médio Risco)'
            elif pd < 0.18: 
                return 'C (Alto Risco)'
            else:
                return 'D (Crítico)'

        df['rating'] = df['pd'].apply(classificar_rating)

        return df