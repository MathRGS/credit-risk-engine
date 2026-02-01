#!/bin/bash

echo "🚧 Iniciando atualização do Portfolio..."

# 1. Se você usar Git no futuro, descomente a linha abaixo:
# git pull origin main

# 2. Derruba o container antigo e sobe o novo reconstruindo a imagem
docker compose up -d --build

# 3. Limpa imagens velhas que sobraram (economiza espaço no disco)
docker image prune -f

echo "✅ Sucesso! O novo Risk Engine já está no ar."