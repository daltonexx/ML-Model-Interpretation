
# ML Model Interpretation

Este repositório contém o código para treinar e avaliar modelos de aprendizado de máquina (KNN, Naïve Bayes e Árvore de Decisão) com interpretação de modelos usando SHAP.

## Passos para rodar o código

### 1. Instalar as dependências

Antes de rodar o código, instale as dependências necessárias com o seguinte comando:

```bash
pip install -r requirements.txt
```

### 2. Rodar o código

Existe duas opções para rodar o código:

#### Opção 1: Rodar com `python3 main.py`(sem gráfico SHAP)

Após a instalação das dependências, você pode rodar o código diretamente com o comando:

```bash
python3 main.py
```

#### Opção 2: Rodar no Jupyter Notebook (com gráfico SHAP)

Se preferir rodar o código interativamente no Jupyter Notebook, siga os passos abaixo.

1. **Iniciar o Jupyter Notebook**:

   - **Em qualquer ambiente local**:

     ```bash
     jupyter notebook
     ```

   - **No GitHub Codespaces ou servidores remotos** (onde não há navegador disponível):

     ```bash
     jupyter notebook --port=8888 --ip=0.0.0.0 --no-browser
     ```

2. **Abrir o Notebook**:

   - Se estiver rodando **localmente**, o Jupyter Notebook será aberto automaticamente no navegador.
   - Se estiver usando **GitHub Codespaces**, o terminal mostrará um **link** como o seguinte:

     ```
     http://127.0.0.1:8888/?token=abc123...
     ```

     Clique nesse link ou copie e cole-o no seu navegador **do Codespace** para acessar o Jupyter Notebook.

3. **Rodar o arquivo `interpretable_model.ipynb`**:

   No Jupyter Notebook, abra o arquivo `interpretable_model.ipynb`. Em seguida, execute as células do notebook interativamente para visualizar os gráficos e resultados.
