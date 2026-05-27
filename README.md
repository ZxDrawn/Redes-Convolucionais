# Clone Didático do CNN Explainer

Este projeto consiste em uma interface web interativa para visualizar as ativações internas de uma Rede Neural Convolucional (CNN), desenhada para fins educacionais.

## Como Executar

Siga os passos abaixo, estritamente nesta ordem, no terminal do seu Mac.

### Parte 1: Treinar e Exportar o Modelo (Backend)

Abra o terminal, navegue até a pasta `backend` do projeto e crie o ambiente virtual:

```bash
cd /Users/karan/Github/cnn-explainer-clone/backend
python3 -m venv venv
```

Ative o ambiente virtual:
```bash
source venv/bin/activate
```

Instale as dependências:
```bash
pip install -r requirements.txt
```

Execute o script de treinamento (este processo fará o download do dataset MNIST, treinará a rede por algumas épocas e exportará o modelo gerado para a pasta `frontend/public/model`):
```bash
python train_and_convert.py
```

### Parte 2: Rodar o Frontend (Interface)

Abra uma **nova aba** no terminal (ou utilize a mesma se desejar, mas lembre-se de ir para a pasta do frontend):

```bash
cd /Users/karan/Github/cnn-explainer-clone/frontend
```

Instale as dependências usando NPM:
```bash
npm install
```

Inicie o servidor de desenvolvimento do Vite:
```bash
npm run dev
```

Abra o navegador no endereço indicado (geralmente `http://localhost:5173`).

---
### Como Utilizar a Interface
1. Clique em **"Carregar Modelo"** para importar os pesos gerados pelo script Python (`model.json` e `.bin`).
2. Uma vez carregado, clique em **"Rodar Inferência (Gato 7)"**. 
3. A imagem de entrada (um dígito 7 sintético) sofrerá um "forward pass" parcial, e você visualizará:
   - A extração das *features* pela camada **Conv2D**.
   - A remoção de valores negativos pela ativação **ReLU**.
   - A redução de dimensionalidade pela **MaxPooling**.
   - As probabilidades numéricas da camada **Dense**.
4. Acompanhe a animação didática de deslize do filtro (um quadrado amarelo simulando o kernel 3x3 correndo a imagem).
