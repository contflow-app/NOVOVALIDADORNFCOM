# Validador NFCom 62 – V2 (Melhorias IA para Relatórios)

Evolução do validador atual, preservando funcionalidades existentes e adicionando:

- Famílias SVA para **relatório fiscal** (EBOOK / LOCAÇÃO / TV/STREAMING / OUTROS)
- Tela de revisão que gera `overrides_sva_familia.yaml` para download/commit
- Ranking de reincidência por cliente
- GPT opcional somente para baixa confiança (ativado no sidebar)

## Rodar local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## GPT (opcional)
No Streamlit Cloud (Settings → Secrets):
```toml
OPENAI_API_KEY="SUA_CHAVE"
```
