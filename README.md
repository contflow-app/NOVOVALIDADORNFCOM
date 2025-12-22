# Validador NFCom 62 + IA (SCM/SVA) – Contare (novo repositório)

Versão separada para evoluir IA e relatórios sem comprometer a versão atual.

## Recursos
- Upload de XML e ZIP com múltiplos XMLs
- Descarte automático de cancelados (evento e protocolo), incluindo CT-e/NF-e/NFCom
- Classificação por IA (OpenAI) para:
  - SCM
  - SVA_EBOOK
  - SVA_LOCACAO
  - SVA_TV_STREAMING
  - SVA_OUTROS
- Correção segura:
  - Remove CFOP apenas quando categoria SVA_* e confiança >= limiar configurado
  - Paliativo de desconto: vProd = vItem quando vProd < vItem (opcional)
- Excel com abas prontas para apuração + histórico mensal por cliente

## Rodar local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud (Secrets)
Em Settings → Secrets:
```toml
OPENAI_API_KEY="SUA_CHAVE"
```

## Logo
Coloque `Logo-Contare-ISP-1.png` na raiz do repositório.

## Histórico mensal
O app grava snapshots em `data/history.db` (SQLite) e permite baixar CSV.
Em Streamlit Community Cloud o armazenamento pode reiniciar — baixe o histórico periodicamente.
