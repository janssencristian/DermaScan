# 🩺 DermaScan – Diagnóstico de Lesões de Pele com IA

Este projeto utiliza **Visão Computacional** e **Redes Neurais Convolucionais (CNNs)** para classificar imagens de lesões dermatológicas. A aplicação é baseada no dataset HAM10000 e usa a arquitetura EfficientNet com explicabilidade via Grad-CAM.

## 🔍 Funcionalidades

- Classificação de 3 tipos de lesões de pele
- Visualização das áreas mais relevantes para a decisão com Grad-CAM
- Interface web com Gradio

## 🚀 Executar Localmente

1. Clone o repositório:
```bash
git clone https://github.com/janssencristian/dermascan.git
cd dermascan

2. Instale as dependências:
pip install -r requirements.txt

3. Execute o app:
python app.py
```

## 💾 Modelo
O arquivo dermascan_model.kegas pode ser baixado manualmente via repositório.

## 🧠 Dataset
Utiliza o Dataset HAM10000 – Skin Lesion Dataset. Disponível em: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
