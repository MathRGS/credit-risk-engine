import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from risk_engine import CreditRiskEngine

st.set_page_config(page_title="Risk Intelligence Dashboard", page_icon="🛡️", layout="wide")

# ------------------ ESTILO VISUAL ------------------
st.markdown("""
<style>
.main { background-color: #f8fafc; }
[data-testid="stMetricValue"] { font-size: 26px; }
div.stButton > button { width: 100%; background-color: #1e293b; color: white; }
</style>
""", unsafe_allow_html=True)

# ------------------ ENGINE ------------------
@st.cache_resource
def load_engine(lgd):
    engine = CreditRiskEngine(lgd=lgd)
    engine.load_model()
    return engine


# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.title("🛡️ Risk Manager")
    st.caption("v2.1 | Production")

    st.divider()
    st.subheader("Parâmetros Globais")

    lgd_param = st.slider(
        "LGD (Loss Given Default)",
        0.1, 1.0, 0.45,
        help="Padrão Basileia: 45% para sênior não garantido"
    )

    engine = load_engine(lgd_param)

    st.divider()

    if st.button("🔄 Regerar Carteira"):
        with st.spinner("Gerando nova carteira com simulação estatística..."):
            st.session_state.portfolio = engine.simulate_portfolio(n_contracts=2000)
        st.success("Carteira regenerada com sucesso!")
        st.rerun()

    st.info("Modelo de PD via Regressão Logística")
    st.markdown("**Matheus Rocha**  \nHead of Risk Analytics")

# ------------------ HEADER ------------------
st.title("Painel de Controle de Risco de Crédito")
st.caption("Monitoramento de exposição, concessão e estresse macroeconômico")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Visão da Carteira",
    "📝 Simulador de Underwriting",
    "⚡ Stress Testing",
    "🔎 Auditoria & Dados"
])

# ------------------ SESSION STATE (EVITA RERUN BUG) ------------------
if "portfolio" not in st.session_state:
    st.session_state.portfolio = engine.simulate_portfolio(2000)

# =========================================================
# 📊 TAB 1 — PORTFOLIO
# =========================================================
with tab1:
    portfolio = st.session_state.portfolio.copy()

    # Atualiza EL caso LGD mude
    portfolio['expected_loss'] = portfolio['pd'] * engine.lgd * portfolio['valor_solicitado']

    # Ordem correta dos ratings
    rating_order = ['A (Baixo Risco)', 'B (Médio Risco)', 'C (Alto Risco)', 'D (Crítico)']
    portfolio['rating'] = pd.Categorical(portfolio['rating'], categories=rating_order, ordered=True)

    total_exposure = portfolio['valor_solicitado'].sum()
    total_el = portfolio['expected_loss'].sum()
    avg_pd = portfolio['pd'].mean()
    risk_cost = (total_el / total_exposure) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exposição Total (EAD)", f"R$ {total_exposure/1e6:.1f} MM")
    c2.metric("Perda Esperada (EL)", f"R$ {total_el/1e6:.2f} MM")
    c3.metric("PD Média", f"{avg_pd:.2%}")
    c4.metric("Custo de Risco", f"{risk_cost:.2f}%")

    st.divider()

    # --- LINHA DE GRÁFICOS ---

    col_vol, col_risk, col_qty = st.columns(3)
    ordem_ratings = ['A (Baixo Risco)', 'B (Médio Risco)', 'C (Alto Risco)', 'D (Crítico)']
    colors_map = {'A (Baixo Risco)': '#10b981', 'B (Médio Risco)': '#3b82f6', 'C (Alto Risco)': '#f59e0b', 'D (Crítico)': '#ef4444'}
    
    config_bloqueada = {'displayModeBar': False, 'scrollZoom': False}

    # GRÁFICO 1: EXPOSIÇÃO
    with col_vol:
        st.subheader("Exposição (Volume)")
        fig_vol = px.bar(
            portfolio.groupby('rating')['valor_solicitado'].sum().reset_index(),
            x='rating', y='valor_solicitado',
            color='rating', text_auto='.2s',
            color_discrete_map=colors_map,
            category_orders={'rating': ordem_ratings}
        )
        fig_vol.update_layout(
            showlegend=False, 
            yaxis_title="R$ Emprestado", 
            xaxis_title=None, 
            height=350,
            xaxis_fixedrange=True, 
            yaxis_fixedrange=True
        )
        st.plotly_chart(fig_vol, use_container_width=True, config=config_bloqueada)

    # GRÁFICO 2: PROVISÃO
    with col_risk:
        st.subheader("Provisão (Perda Esp.)")
        fig_risk = px.bar(
            portfolio.groupby('rating')['expected_loss'].sum().reset_index(),
            x='rating', y='expected_loss',
            color='rating', text_auto='.2s',
            color_discrete_map=colors_map,
            category_orders={'rating': ordem_ratings}
        )
        fig_risk.update_layout(
            showlegend=False, 
            yaxis_title="R$ Provisão", 
            xaxis_title=None, 
            height=350,
            xaxis_fixedrange=True, 
            yaxis_fixedrange=True
        )
        st.plotly_chart(fig_risk, use_container_width=True, config=config_bloqueada)
    
    # GRÁFICO 3: QUALIDADE
    with col_qty:
        st.subheader("Mix da Carteira")
        fig_pie = px.pie(
            portfolio, names='rating', hole=0.5,
            color='rating',
            color_discrete_map=colors_map,
            category_orders={'rating': ordem_ratings}
        )
        fig_pie.update_layout(height=350, showlegend=True, legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_pie, use_container_width=True, config=config_bloqueada)

    st.markdown("*Para alterar o LGD global, utilize o controle na barra lateral esquerda.*")
# =========================================================
# 📝 TAB 2 — UNDERWRITING
# =========================================================
with tab2:
    st.subheader("Simulação de Crédito Individual")

    col_in, col_out = st.columns([1, 2])

    with col_in:
        with st.form("form"):
            renda = st.number_input(
                "Renda Mensal",
                min_value=1500.0,
                max_value=100000.0,
                value=5000.0,
                step=500.0,          
                format="%.2f"
            )
            divida = st.number_input(
                "Dívida Atual",
                min_value=0.0,
                max_value=2000000.0,
                value=2000.0,
                step=500.0,
                format="%.2f"
            )
            score = st.slider("Score Bureau", 0, 1000, 650)
            idade = st.number_input("Idade", 18, 90, 35)
            valor = st.number_input(
                "Valor Solicitado",
                min_value=1000.0,
                max_value=5000000.0,
                value=30000.0,
                step=10000.0,         
                format="%.2f"
            )
            submit = st.form_submit_button("Calcular Risco")

    with col_out:
        if submit:
            df = pd.DataFrame({
                'renda_mensal': [renda],
                'score_serasa': [score],
                'idade': [idade],
                'divida_total': [divida]
            })

            pd_cliente = engine.predict(df)[0]
            el_cliente = pd_cliente * engine.lgd * valor

            k1, k2, k3 = st.columns(3)
            k1.metric("PD", f"{pd_cliente:.2%}")
            k2.metric("Rating", "A" if pd_cliente < 0.1 else "B" if pd_cliente < 0.3 else "C")
            k3.metric("Perda Esperada", f"R$ {el_cliente:,.2f}")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pd_cliente * 100,
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 10], 'color': "#10b981"},
                        {'range': [10, 30], 'color': "#f59e0b"},
                        {'range': [30, 100], 'color': "#ef4444"}
                    ]
                },
                title={'text': "Risco de Inadimplência (%)"}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

# =========================================================
# ⚡ TAB 3 — STRESS TEST
# =========================================================
with tab3:
    st.subheader("Stress Test Macroeconômico")

    scenario = st.radio("Cenário", ["Base", "Recessão Leve (+20% PD)", "Crise Severa (+50% PD)"])

    factor = 1.0 if scenario == "Base" else 1.2 if "20%" in scenario else 1.5

    if st.button("Executar Stress Test"):
        base = st.session_state.portfolio.copy()
        base['expected_loss'] = base['pd'] * engine.lgd * base['valor_solicitado']

        stress = base.copy()
        stress['pd'] = (stress['pd'] * factor).clip(upper=1)
        stress['expected_loss'] = stress['pd'] * engine.lgd * stress['valor_solicitado']

        el_base = base['expected_loss'].sum()
        el_stress = stress['expected_loss'].sum()
        delta = el_stress - el_base

        m1, m2 = st.columns(2)
        m1.metric("EL Base", f"R$ {el_base/1e6:.2f} MM")
        m2.metric("EL Estressado", f"R$ {el_stress/1e6:.2f} MM", delta=f"+R$ {delta/1e6:.2f} MM")

        df_chart = pd.DataFrame({"Cenário": ["Base", "Estressado"], "EL": [el_base, el_stress]})
        fig = px.bar(df_chart, x="Cenário", y="EL", color="Cenário",
                     color_discrete_sequence=["#3b82f6", "#ef4444"])
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 4: AUDITORIA & DADOS ---
with tab4:
    st.subheader("Auditoria da Carteira e Transparência do Modelo")
    portfolio = st.session_state.portfolio.copy()

    st.markdown("### 📌 Estatísticas Gerais da Base")
    colA, colB, colC = st.columns(3)
    colA.metric("Contratos", len(portfolio))
    colB.metric("PD Média", f"{portfolio['pd'].mean():.2%}")
    colC.metric("Loss Given Default Médio", f"{portfolio['loss_given_default'].mean():.0%}")

    st.markdown("---")
    st.markdown("### 📊 Distribuições Estatísticas")

    c1, c2 = st.columns(2)

    with c1:
        fig_pd = px.histogram(portfolio, x='pd', nbins=30, title="Distribuição de Probabilidade de Default (PD)")
        st.plotly_chart(fig_pd, use_container_width=True)

    with c2:
        fig_el = px.histogram(portfolio, x='expected_loss', nbins=30, title="Distribuição de Perda Esperada (EL)")
        st.plotly_chart(fig_el, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧮 Estatísticas Descritivas")
    st.dataframe(portfolio.describe(), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔍 Base Completa (Auditoria de Contratos)")
    st.caption("Use filtros e busca para auditoria individual de operações")

    st.dataframe(
        portfolio.sort_values("expected_loss", ascending=False),
        use_container_width=True,
        height=400
    )

    csv = portfolio.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Baixar carteira completa (CSV)",
        data=csv,
        file_name="carteira_risco_credito.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("© 2026 Matheus Rocha — Credit Risk Intelligence Platform")
