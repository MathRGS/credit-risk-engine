# Usei a 3.10 que é mais estável para as libs novas que usamos
FROM python:3.10-slim

WORKDIR /app

# 1. Instalação do Sistema (Raramente muda, fica no cache)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- A MÁGICA DO CACHE ---

# 2. Copia APENAS o arquivo de requisitos primeiro
COPY requirements.txt .

# 3. Instala as bibliotecas Python
# Se você mudar o app.py, o Docker VÊ que o requirements.txt NÃO mudou.
# Então ele PULA essa etapa (usa o cache) e ganha tempo.
RUN pip install --no-cache-dir -r requirements.txt

# 4. Só AGORA copia o restante dos arquivos (app.py, risk_engine.py)
# Essa é a única camada que será refeita nas suas atualizações diárias.
COPY . .

# -------------------------

EXPOSE 8501

# Comandos de segurança para produção
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]