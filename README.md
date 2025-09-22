# PathFinder for FamilySearch

PathFinder é uma ferramenta web para descobrir e visualizar todas as ligações de parentesco entre duas pessoas na Árvore Familiar do FamilySearch.

Enquanto a ferramenta nativa geralmente mostra apenas um caminho, o PathFinder realiza uma busca mais ampla para revelar múltiplas conexões, úteis em casos de endogamia ou relações complexas.

**Status:** Versão Beta.

## Configuração (Desenvolvimento Local)

1.  Clone o repositório.
2.  Crie um ambiente virtual: `python -m venv venv`
3.  Ative o ambiente e instale as dependências: `pip install -r requirements.txt`
4.  Crie um arquivo `.env` e preencha as seguintes variáveis:
    ```
    FAMILYSEARCH_APP_KEY=SUA_CHAVE_DE_APP
    FAMILYSEARCH_REDIRECT_URI=[https://127.0.0.1:5000/callback](https://127.0.0.1:5000/callback)
    SECRET_KEY=UMA_CHAVE_SECRETA_ALEATORIA_LONGA
    FLASK_DEBUG=1
    ```
5.  Execute a aplicação: `python flask_app.py`
