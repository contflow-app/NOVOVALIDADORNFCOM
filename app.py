
import io
import os
import re
import json
import zipfile
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd
import streamlit as st
from lxml import etree

# Optional dependency: OpenAI (enabled if OPENAI_API_KEY present)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Optional dependency: YAML configs
try:
    import yaml
except Exception:
    yaml = None

# Optional dependency: PDF
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import simpleSplit
except Exception:
    A4 = None
    canvas = None
    simpleSplit = None

APP_TITLE = "Validador NFCom 62 + IA (SCM/SVA) – Contare"
LOGO_PATH = "Logo-Contare-ISP-1.png"   # coloque no repo na raiz
DB_PATH = os.path.join("data", "history.db")

st.set_page_config(page_title=APP_TITLE, layout="wide")

# =========================================================
# Utilities
# =========================================================

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    trans = str.maketrans({
        "á":"a","à":"a","ã":"a","â":"a",
        "é":"e","ê":"e",
        "í":"i",
        "ó":"o","õ":"o","ô":"o",
        "ú":"u",
        "ç":"c",
    })
    s = s.translate(trans)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def to_float(x) -> float:
    try:
        if x is None:
            return 0.0
        return float(str(x).replace(",", "."))
    except Exception:
        return 0.0

def num_to_br(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        x = float(value)
        s = f"{x:,.2f}"
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
        return s
    except Exception:
        return str(value)

# =========================================================
# XML / Namespace helpers
# =========================================================

def parse_xml(file_bytes: bytes) -> etree._ElementTree:
    parser = etree.XMLParser(remove_blank_text=True, recover=True)
    try:
        return etree.parse(io.BytesIO(file_bytes), parser)
    except Exception as e:
        raise ValueError(f"XML inválido: {e}")

def get_ns(tree: etree._ElementTree) -> Dict[str, str]:
    root = tree.getroot()
    default_ns = root.nsmap.get(None)
    return {"n": default_ns} if default_ns else {}

def xp(node, ns, expr: str):
    if ns:
        try:
            return node.xpath(expr, namespaces=ns)
        except Exception:
            return node.xpath(expr)
    return node.xpath(expr)

def first_text(node, ns, expr: str) -> str:
    nodes = xp(node, ns, expr)
    if not nodes:
        return ""
    n = nodes[0]
    if isinstance(n, etree._Element):
        return (n.text or "").strip()
    return str(n).strip()

def extract_chave_acesso(tree: etree._ElementTree) -> str:
    root = tree.getroot()
    ns = get_ns(tree)
    for path in [
        ".//n:infNFCom/@Id", ".//infNFCom/@Id",
        ".//n:infNFe/@Id", ".//infNFe/@Id",
        ".//n:infCte/@Id", ".//infCte/@Id",
    ]:
        ids = xp(root, ns, path)
        if ids:
            m = re.search(r"\d{44}", str(ids[0]))
            if m:
                return m.group(0)
    xml_str = etree.tostring(root, encoding="unicode")
    m2 = re.search(r"\d{44}", xml_str)
    return m2.group(0) if m2 else ""

def get_nf_model(tree: etree._ElementTree) -> str:
    root = tree.getroot()
    ns = get_ns(tree)
    mod = first_text(root, ns, ".//n:ide/n:mod | .//ide/mod")
    return (mod or "").strip()

def get_emitente(tree: etree._ElementTree) -> Tuple[str, str]:
    root = tree.getroot()
    ns = get_ns(tree)
    cnpj = first_text(root, ns, ".//n:emit/n:CNPJ | .//emit/CNPJ").strip()
    xnome = first_text(root, ns, ".//n:emit/n:xNome | .//emit/xNome").strip()
    return (cnpj, xnome)

def get_competencia_mes(tree: etree._ElementTree) -> str:
    root = tree.getroot()
    ns = get_ns(tree)
    comp = first_text(root, ns, ".//n:gFat/n:CompetFat | .//gFat/CompetFat").strip()
    if comp:
        m = re.search(r"(\d{4})[-/]?(\d{2})", comp)
        if m:
            return f"{m.group(1)}-{m.group(2)}"
        return comp[:7]
    demi = first_text(root, ns, ".//n:ide/n:dEmi | .//ide/dEmi").strip()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", demi)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return datetime.now().strftime("%Y-%m")

# =========================================================
# Cancelamento detection (evento + protocolo) – NFCom/NFe/CTe
# =========================================================

def contains_cancel_words(text: str) -> bool:
    t = normalize_text(text or "")
    return ("cancelamento" in t) or ("cancelad" in t)

def detect_cancelamento_event_bytes(xml_bytes: bytes) -> Tuple[bool, Optional[str], Optional[str]]:
    try:
        tree = parse_xml(xml_bytes)
    except Exception:
        return (False, None, None)

    root = tree.getroot()
    ns = get_ns(tree)

    tp = first_text(root, ns, ".//n:tpEvento | .//tpEvento")
    if tp != "110111":
        return (False, None, None)

    xevt = first_text(root, ns, ".//n:xEvento | .//xEvento")
    xmot = first_text(root, ns, ".//n:xMotivo | .//xMotivo")
    if not (contains_cancel_words(xevt) or contains_cancel_words(xmot)):
        return (False, None, None)

    ch_nfcom = first_text(root, ns, ".//n:chNFCom | .//chNFCom")
    if ch_nfcom:
        return (True, ch_nfcom, "NFCom")
    ch_nfe = first_text(root, ns, ".//n:chNFe | .//chNFe")
    if ch_nfe:
        return (True, ch_nfe, "NFe")
    ch_cte = first_text(root, ns, ".//n:chCTe | .//chCTe")
    if ch_cte:
        return (True, ch_cte, "CTe")

    xml_str = etree.tostring(root, encoding="unicode")
    m = re.search(r"\d{44}", xml_str)
    return (True, m.group(0) if m else None, "desconhecido")

def detect_canceled_by_protocol_bytes(xml_bytes: bytes) -> Tuple[bool, Optional[str], Optional[str]]:
    try:
        tree = parse_xml(xml_bytes)
    except Exception:
        return (False, None, None)
    root = tree.getroot()
    ns = get_ns(tree)

    textos: List[str] = []
    for n in xp(root, ns, ".//n:xMotivo | .//xMotivo"):
        if isinstance(n, etree._Element) and n.text:
            textos.append(n.text)
    for n in xp(root, ns, ".//n:xEvento | .//xEvento"):
        if isinstance(n, etree._Element) and n.text:
            textos.append(n.text)

    if not any(contains_cancel_words(t) for t in textos):
        return (False, None, None)

    chave = extract_chave_acesso(tree)

    tipo = "desconhecido"
    if xp(root, ns, ".//n:infCte | .//infCte"):
        tipo = "CTe"
    elif xp(root, ns, ".//n:infNFe | .//infNFe"):
        tipo = "NFe"
    elif xp(root, ns, ".//n:infNFCom | .//infNFCom"):
        tipo = "NFCom"

    return (True, chave, tipo)

# =========================================================
# Configs (rules.yaml + cclass_config.yaml)
# =========================================================

@st.cache_data
def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

@st.cache_data
def load_rules(path: str = "rules.yaml") -> List[Dict[str, Any]]:
    cfg = load_yaml(path)
    if isinstance(cfg, list):
        return cfg
    if isinstance(cfg, dict):
        return cfg.get("rules", []) or []
    return []

@st.cache_data
def load_cclass_config(path: str = "cclass_config.yaml") -> Dict[str, Any]:
    cfg = load_yaml(path)
    return cfg if isinstance(cfg, dict) else {}

# =========================================================
# Heuristic classification (strong-first)
# =========================================================

SCM_KEYWORDS = [
    "fibra", "fibra optica", "fibra óptica", "banda larga", "internet", "link",
    "link dedicado", "dedicado", "ftth", "plano", "velocidade", "scm", "dados",
    "conexao", "conexão", "wifi", "wi-fi", "provedor", "acesso", "rede", "conectividade"
]
SVA_EBOOK_KEYWORDS = ["ebook", "e-book", "livro digital", "biblioteca digital", "leitura", "plataforma de leitura"]
SVA_LOCACAO_KEYWORDS = ["locacao", "locação", "comodato", "aluguel", "locar", "equipamento", "roteador", "onu", "cpe"]
SVA_TV_KEYWORDS = ["tv", "iptv", "streaming", "conteudo", "conteúdo", "televisao", "televisão"]
SVA_GENERIC_KEYWORDS = [
    "antivirus", "anti-virus", "anti vírus", "antivírus", "email", "e-mail",
    "ip fixo", "backup", "suporte premium", "voip", "telefonia",
    "servico adicional", "serviço adicional", "servicos adicionais", "serviços adicionais", "sva",
    "cloud", "nuvem", "seguranca", "segurança"
]

def heuristic_category(desc: str) -> Tuple[str, float, str]:
    d = normalize_text(desc)
    if not d:
        return ("SVA_OUTROS", 0.50, "Descrição vazia")

    # strong SCM
    if any(k in d for k in SCM_KEYWORDS) and not any(k in d for k in SVA_GENERIC_KEYWORDS):
        return ("SCM", 0.96, "Palavras-chave fortes de SCM")

    # SVA subcats
    if any(k in d for k in SVA_EBOOK_KEYWORDS) and not any(k in d for k in SCM_KEYWORDS):
        return ("SVA_EBOOK", 0.96, "Palavras-chave eBook")
    if any(k in d for k in SVA_LOCACAO_KEYWORDS) and not any(k in d for k in SCM_KEYWORDS):
        return ("SVA_LOCACAO", 0.95, "Palavras-chave locação/equipamento")
    if any(k in d for k in SVA_TV_KEYWORDS) and not any(k in d for k in SCM_KEYWORDS):
        return ("SVA_TV_STREAMING", 0.95, "Palavras-chave TV/Streaming")

    if any(k in d for k in SVA_GENERIC_KEYWORDS) and not any(k in d for k in SCM_KEYWORDS):
        return ("SVA_OUTROS", 0.90, "Palavras-chave SVA (genérico)")

    # ambiguous
    if any(k in d for k in SCM_KEYWORDS) and any(k in d for k in SVA_GENERIC_KEYWORDS):
        return ("SCM", 0.70, "Ambíguo (SCM+SVA). Revisar.")

    return ("SVA_OUTROS", 0.60, "Sem evidência forte. IA/Revisão.")

# =========================================================
# OpenAI classification (only when needed)
# =========================================================

AI_ALLOWED = {"SCM", "SVA_EBOOK", "SVA_LOCACAO", "SVA_TV_STREAMING", "SVA_OUTROS"}

AI_SYSTEM = """Você é um classificador fiscal para itens de NFCom (Modelo 62).
Classifique cada item em UMA categoria:
- SCM
- SVA_EBOOK
- SVA_LOCACAO
- SVA_TV_STREAMING
- SVA_OUTROS

Regras:
- Internet/fibra/link/plano => SCM.
- eBook/leitura/biblioteca => SVA_EBOOK.
- locação/aluguel/comodato/equipamento => SVA_LOCACAO.
- TV/streaming/IPTV/conteúdo => SVA_TV_STREAMING.
- Se sem evidência suficiente => SVA_OUTROS com baixa confiança.
Retorne SOMENTE JSON válido.
"""

def get_openai_client():
    key = None
    if "OPENAI_API_KEY" in st.secrets:
        key = st.secrets["OPENAI_API_KEY"]
    key = key or os.environ.get("OPENAI_API_KEY")
    if not key or OpenAI is None:
        return None
    return OpenAI(api_key=key)

def ai_classify_batch(items: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    client = get_openai_client()
    if client is None:
        out = []
        for it in items:
            cat, conf, why = heuristic_category(it.get("desc", ""))
            out.append({
                "id": it.get("id"),
                "categoria_fiscal_ia": cat,
                "confianca_ia": float(conf),
                "motivo_ia": f"Sem OpenAI. Heurística: {why}",
                "recomendacao_cfop": "MANTER" if cat == "SCM" else "REMOVER",
                "origem": "heuristica"
            })
        return out

    payload = [{
        "id": it.get("id"),
        "descricao": (it.get("desc") or "")[:220],
        "cClass": (it.get("cClass") or "")[:32],
        "CFOP": (it.get("cfop") or "")[:16],
    } for it in items]

    user_msg = {
        "taxonomy": sorted(list(AI_ALLOWED)),
        "items": payload,
        "output_schema": {
            "items": [{
                "id": "string",
                "categoria_fiscal_ia": "SCM|SVA_EBOOK|SVA_LOCACAO|SVA_TV_STREAMING|SVA_OUTROS",
                "confianca_ia": "number between 0 and 1",
                "motivo_ia": "short string",
                "recomendacao_cfop": "MANTER|REMOVER"
            }]
        }
    }

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": AI_SYSTEM},
            {"role": "user", "content": "Classifique os itens abaixo. Retorne SOMENTE JSON.\n\n" + json.dumps(user_msg, ensure_ascii=False)}
        ],
        temperature=0.0,
        max_output_tokens=1200
    )

    text = resp.output_text
    try:
        data = json.loads(text)
        results = data.get("items", [])
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        results = []
        if m:
            data = json.loads(m.group(0))
            results = data.get("items", [])

    by_id = {r.get("id"): r for r in results if isinstance(r, dict)}
    out = []
    for it in items:
        r = by_id.get(it.get("id"))
        if not r:
            cat, conf, why = heuristic_category(it.get("desc",""))
            out.append({
                "id": it.get("id"),
                "categoria_fiscal_ia": cat,
                "confianca_ia": float(conf),
                "motivo_ia": f"IA falhou. Heurística: {why}",
                "recomendacao_cfop": "MANTER" if cat == "SCM" else "REMOVER",
                "origem": "fallback_heuristica"
            })
            continue
        cat = r.get("categoria_fiscal_ia","SVA_OUTROS")
        if cat not in AI_ALLOWED:
            cat = "SVA_OUTROS"
        conf = float(r.get("confianca_ia", 0.6) or 0.6)
        conf = max(0.0, min(1.0, conf))
        out.append({
            "id": it.get("id"),
            "categoria_fiscal_ia": cat,
            "confianca_ia": conf,
            "motivo_ia": (r.get("motivo_ia") or "").strip()[:220],
            "recomendacao_cfop": "MANTER" if cat == "SCM" else "REMOVER",
            "origem": "openai"
        })
    return out

# =========================================================
# Rules engine (minimal)
# =========================================================

def apply_rules_yaml(tree: etree._ElementTree, rules: List[Dict[str, Any]], file_name: str) -> List[Dict[str, Any]]:
    errors = []
    root = tree.getroot()
    ns = get_ns(tree)

    for rule in rules:
        rid = rule.get("id", "R")
        tipo = rule.get("tipo")
        expr = rule.get("xpath", "")
        found = xp(root, ns, expr) if expr else []

        if tipo == "obrigatorio":
            if not found:
                errors.append({
                    "arquivo": file_name,
                    "regra_id": rid,
                    "descricao_regra": rule.get("descricao", ""),
                    "campo_xpath": expr,
                    "valor_encontrado": "",
                    "mensagem_erro": rule.get("mensagem_erro", "Campo obrigatório ausente."),
                    "sugestao_correcao": rule.get("sugestao_correcao", "Preencher o campo."),
                    "nivel": rule.get("nivel", "erro"),
                })
            else:
                for n in found:
                    txt = (n.text or "").strip() if isinstance(n, etree._Element) else str(n).strip()
                    if not txt:
                        errors.append({
                            "arquivo": file_name,
                            "regra_id": rid,
                            "descricao_regra": rule.get("descricao", ""),
                            "campo_xpath": expr,
                            "valor_encontrado": txt,
                            "mensagem_erro": rule.get("mensagem_erro", "Campo obrigatório vazio."),
                            "sugestao_correcao": rule.get("sugestao_correcao", "Preencher o campo."),
                            "nivel": rule.get("nivel", "erro"),
                        })
    return errors

# =========================================================
# Item extraction + correction
# =========================================================

def extract_items_nfcom(tree: etree._ElementTree, file_name: str) -> List[Dict[str, Any]]:
    root = tree.getroot()
    ns = get_ns(tree)
    dets = xp(root, ns, ".//n:det | .//det")
    items = []
    for idx, det in enumerate(dets, start=1):
        cclass = first_text(det, ns, "./n:prod/n:cClass | ./prod/cClass")
        xprod = first_text(det, ns, "./n:prod/n:xProd | ./prod/xProd")
        cfop  = first_text(det, ns, "./n:prod/n:CFOP | ./prod/CFOP")
        qfat  = first_text(det, ns, "./n:prod/n:qFaturada | ./prod/qFaturada")
        umed  = first_text(det, ns, "./n:prod/n:uMed | ./prod/uMed")

        vitem = to_float(first_text(det, ns, "./n:prod/n:vItem | ./prod/vItem"))
        vprod = to_float(first_text(det, ns, "./n:prod/n:vProd | ./prod/vProd"))
        vdesc = to_float(first_text(det, ns, "./n:prod/n:vDesc | ./prod/vDesc"))
        vout  = to_float(first_text(det, ns, "./n:prod/n:vOutro | ./prod/vOutro"))

        vbcicms = to_float(first_text(det, ns, "./n:imposto/n:ICMS/n:vBC | ./imposto/ICMS/vBC"))
        picms   = to_float(first_text(det, ns, "./n:imposto/n:ICMS/n:pICMS | ./imposto/ICMS/pICMS"))
        vicms   = to_float(first_text(det, ns, "./n:imposto/n:ICMS/n:vICMS | ./imposto/ICMS/vICMS"))

        items.append({
            "arquivo": file_name,
            "item": idx,
            "cClass": cclass,
            "descricao": xprod,
            "CFOP": cfop,
            "qFaturada": qfat,
            "uMed": umed,
            "vItem": vitem,
            "vProd": vprod,
            "vDesc": vdesc,
            "vOutros": vout,
            "vServ": vprod,
            "vBCICMS": vbcicms,
            "pICMS": picms,
            "vICMS": vicms,
        })
    return items

def correct_xml_nfcom(tree: etree._ElementTree, df_dec: pd.DataFrame, corr_auto_threshold: float, corrigir_descontos: bool) -> bytes:
    root = tree.getroot()
    copy_root = etree.fromstring(etree.tostring(root))
    new_tree = etree.ElementTree(copy_root)
    ns = get_ns(new_tree)

    decisions = {int(r["item"]): (str(r["categoria_fiscal_ia"]), float(r["confianca_ia"])) for _, r in df_dec.iterrows()}

    dets = xp(copy_root, ns, ".//n:det | .//det")
    for idx, det in enumerate(dets, start=1):
        cat, conf = decisions.get(idx, ("SVA_OUTROS", 0.0))

        if cat.startswith("SVA_") and conf >= corr_auto_threshold:
            cfop_nodes = xp(det, ns, "./n:prod/n:CFOP | ./prod/CFOP")
            for node in cfop_nodes:
                parent = node.getparent()
                if parent is not None:
                    parent.remove(node)

        if corrigir_descontos:
            vitem_nodes = xp(det, ns, "./n:prod/n:vItem | ./prod/vItem")
            vprod_nodes = xp(det, ns, "./n:prod/n:vProd | ./prod/vProd")
            if vitem_nodes and vprod_nodes:
                vi_text = (vitem_nodes[0].text or "").strip()
                vp_text = (vprod_nodes[0].text or "").strip()
                vi = to_float(vi_text)
                vp = to_float(vp_text)
                if vp < vi:
                    vprod_nodes[0].text = vi_text

    return etree.tostring(new_tree, encoding="utf-8", xml_declaration=True)

# =========================================================
# History DB
# =========================================================

def db_connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def db_init():
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS client_month (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            month TEXT NOT NULL,
            emit_cnpj TEXT NOT NULL,
            emit_nome TEXT,
            total_docs_ativos INTEGER,
            total_docs_cancelados INTEGER,
            total_itens INTEGER,
            total_erros INTEGER,
            total_vserv REAL,
            total_scm REAL,
            total_sva_ebook REAL,
            total_sva_locacao REAL,
            total_sva_tv REAL,
            total_sva_outros REAL,
            itens_revisar INTEGER,
            created_at TEXT
        )
    """)
    con.commit()
    con.close()

def db_insert_snapshot(row: Dict[str, Any]):
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO client_month (
            month, emit_cnpj, emit_nome,
            total_docs_ativos, total_docs_cancelados, total_itens, total_erros, total_vserv,
            total_scm, total_sva_ebook, total_sva_locacao, total_sva_tv, total_sva_outros,
            itens_revisar, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row.get("month"), row.get("emit_cnpj"), row.get("emit_nome"),
        int(row.get("total_docs_ativos", 0)), int(row.get("total_docs_cancelados", 0)),
        int(row.get("total_itens", 0)), int(row.get("total_erros", 0)),
        float(row.get("total_vserv", 0.0)),
        float(row.get("total_scm", 0.0)),
        float(row.get("total_sva_ebook", 0.0)),
        float(row.get("total_sva_locacao", 0.0)),
        float(row.get("total_sva_tv", 0.0)),
        float(row.get("total_sva_outros", 0.0)),
        int(row.get("itens_revisar", 0)),
        row.get("created_at")
    ))
    con.commit()
    con.close()

def db_load_history(emit_cnpj: Optional[str] = None) -> pd.DataFrame:
    con = db_connect()
    q = "SELECT * FROM client_month"
    params = ()
    if emit_cnpj:
        q += " WHERE emit_cnpj = ?"
        params = (emit_cnpj,)
    df = pd.read_sql_query(q, con, params=params)
    con.close()
    return df

# =========================================================
# Excel report
# =========================================================

def generate_excel_report(
    df_resumo: pd.DataFrame,
    df_erros: pd.DataFrame,
    df_consolidado: pd.DataFrame,
    df_detalhe: pd.DataFrame,
    df_categoria: pd.DataFrame,
    df_categoria_arquivo: pd.DataFrame,
    df_revisar: pd.DataFrame,
    df_status: pd.DataFrame,
    df_snapshot: pd.DataFrame,
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_resumo.to_excel(writer, sheet_name="Resumo", index=False)
        if not df_erros.empty:
            df_erros.to_excel(writer, sheet_name="Erros Detalhados", index=False)
        if not df_consolidado.empty:
            df_consolidado.to_excel(writer, sheet_name="Erros Consolidados", index=False)
        if not df_detalhe.empty:
            df_detalhe.to_excel(writer, sheet_name="Detalhamento Itens", index=False)
        if not df_categoria.empty:
            df_categoria.to_excel(writer, sheet_name="Resumo_Categoria_IA", index=False)
        if not df_categoria_arquivo.empty:
            df_categoria_arquivo.to_excel(writer, sheet_name="Categoria_por_Arquivo", index=False)
        if not df_revisar.empty:
            df_revisar.to_excel(writer, sheet_name="Itens_Revisar", index=False)
        if not df_status.empty:
            df_status.to_excel(writer, sheet_name="Status_XMLs", index=False)
        if not df_snapshot.empty:
            df_snapshot.to_excel(writer, sheet_name="Historico_Snapshot", index=False)
    output.seek(0)
    return output.read()

# =========================================================
# Main UI
# =========================================================

def main():
    db_init()

    c1, c2 = st.columns([1, 4])
    with c1:
        try:
            st.image(LOGO_PATH)
        except Exception:
            st.write("")
    with c2:
        st.markdown(f"## {APP_TITLE}")
        st.caption("Desenvolvido por Raul Martins — Contare Contabilidade especializada em Provedores de Internet")

    st.write("Nova versão em repositório separado (segura) para evoluir IA e relatórios sem impactar o validador atual.")

    st.sidebar.header("Configurações")
    consolidar = st.sidebar.checkbox("Consolidar erros iguais", value=True)
    corrigir_descontos = st.sidebar.checkbox("Paliativo descontos: vProd = vItem quando vProd < vItem", value=False)

    enable_ai = st.sidebar.checkbox("Ativar IA (OpenAI) para classificação", value=True)
    ai_model = st.sidebar.text_input("Modelo OpenAI", value="gpt-4o-mini")
    corr_auto_threshold = st.sidebar.slider("Limiar para correção automática (SVA remove CFOP)", 0.50, 1.00, 0.95, 0.01)
    suggest_threshold = st.sidebar.slider("Limiar para sugestão forte", 0.50, 1.00, 0.85, 0.01)

    st.sidebar.markdown("---")
    cancel_file = st.sidebar.file_uploader("Chaves canceladas (CSV/TXT 44 dígitos)", type=["csv","txt"])
    cancel_keys = set()
    if cancel_file is not None:
        raw = cancel_file.read()
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            text = raw.decode("latin1", errors="ignore")
        cancel_keys = set(re.findall(r"\d{44}", text))

    uploaded = st.file_uploader("Envie XML (NFCom) ou ZIP com XMLs", type=["xml","zip"], accept_multiple_files=True)

    rules = load_rules("rules.yaml")

    if uploaded and st.button("Processar lote"):
        errors_all: List[Dict[str, Any]] = []
        invalid: List[Dict[str, Any]] = []
        items_all: List[Dict[str, Any]] = []
        xml_ativos: List[Dict[str, Any]] = []
        canceled: List[Dict[str, Any]] = []

        client_cnpj = None
        client_nome = None
        month_ref = None

        def handle_xml(xml_bytes: bytes, base_name: str, logical_name: str):
            nonlocal client_cnpj, client_nome, month_ref

            is_evt, chave_evt, tipo_evt = detect_cancelamento_event_bytes(xml_bytes)
            if is_evt:
                canceled.append({"arquivo_base": base_name, "chave": chave_evt, "status": f"evento_cancelamento_{tipo_evt}"})
                return

            is_prot, chave_prot, tipo_prot = detect_canceled_by_protocol_bytes(xml_bytes)
            if is_prot:
                canceled.append({"arquivo_base": base_name, "chave": chave_prot, "status": f"cancelado_protocolo_{tipo_prot}"})
                return

            tree = parse_xml(xml_bytes)
            model = get_nf_model(tree)
            if model and model != "62":
                invalid.append({"arquivo": logical_name, "erro": f"Modelo {model} != 62 (ignorado)."})
                return

            chave = extract_chave_acesso(tree)
            if cancel_keys and chave and chave in cancel_keys:
                canceled.append({"arquivo_base": base_name, "chave": chave, "status": "lista_canceladas"})
                return

            if client_cnpj is None:
                client_cnpj, client_nome = get_emitente(tree)
            if month_ref is None:
                month_ref = get_competencia_mes(tree)

            if rules:
                errors_all.extend(apply_rules_yaml(tree, rules, logical_name))

            items_all.extend(extract_items_nfcom(tree, logical_name))

            xml_ativos.append({
                "base_name": base_name,
                "logical_name": logical_name,
                "tree": tree,
                "chave": chave
            })

        for f in uploaded:
            fname = f.name
            content = f.read()
            if fname.lower().endswith(".zip"):
                try:
                    with zipfile.ZipFile(io.BytesIO(content)) as zf:
                        for info in zf.infolist():
                            if info.filename.lower().endswith(".xml"):
                                base_name = info.filename.replace("\\", "/").replace("/", "_")
                                xml_bytes = zf.read(info)
                                handle_xml(xml_bytes, base_name, f"{fname}::{info.filename}")
                except zipfile.BadZipFile:
                    invalid.append({"arquivo": fname, "erro": "ZIP inválido/corrompido."})
            else:
                handle_xml(content, fname, fname)

        df_invalid = pd.DataFrame(invalid) if invalid else pd.DataFrame()
        df_errors = pd.DataFrame(errors_all) if errors_all else pd.DataFrame()
        df_items = pd.DataFrame(items_all) if items_all else pd.DataFrame()

        if not xml_ativos or df_items.empty:
            st.warning("Nenhum XML NFCom ativo foi processado.")
            if not df_invalid.empty:
                st.subheader("Ignorados")
                st.dataframe(df_invalid, use_container_width=True)
            if canceled:
                st.subheader("Cancelados")
                st.dataframe(pd.DataFrame(canceled), use_container_width=True)
            return

        # IA stage
        df_items["desc_norm"] = df_items["descricao"].fillna("").map(normalize_text)
        df_items["id_desc"] = df_items["desc_norm"].map(lambda x: re.sub(r"[^a-z0-9]+", "_", x)[:80])

        heur = df_items["descricao"].fillna("").map(lambda d: heuristic_category(d))
        df_items["categoria_heur"], df_items["confianca_heur"], df_items["motivo_heur"] = zip(*heur)

        need_ai_mask = (enable_ai) & (df_items["confianca_heur"].astype(float) < suggest_threshold)
        df_need = df_items.loc[need_ai_mask, ["id_desc","descricao","cClass","CFOP"]].drop_duplicates("id_desc").copy()

        ai_by_id: Dict[str, Dict[str, Any]] = {}
        if enable_ai and not df_need.empty:
            st.info(f"IA: classificando {len(df_need)} descrições (deduplicadas)…")
            batch = [{"id": r["id_desc"], "desc": r["descricao"], "cClass": r.get("cClass",""), "cfop": r.get("CFOP","")} for _, r in df_need.iterrows()]
            chunk_size = 50
            for i in range(0, len(batch), chunk_size):
                chunk = batch[i:i+chunk_size]
                res = ai_classify_batch(chunk, model=ai_model)
                for r in res:
                    ai_by_id[r["id"]] = r

        def decide_row(r):
            cat = r["categoria_heur"]
            conf = float(r["confianca_heur"])
            motivo = r["motivo_heur"]
            origem = "heuristica"
            ai = ai_by_id.get(r["id_desc"])
            if ai is not None and conf < 0.95:
                cat = ai.get("categoria_fiscal_ia", cat)
                conf = float(ai.get("confianca_ia", conf))
                motivo = ai.get("motivo_ia", motivo)
                origem = ai.get("origem", "openai")
            if cat not in AI_ALLOWED:
                cat = "SVA_OUTROS"
            cfop_action = "MANTER" if cat == "SCM" else "REMOVER"
            if conf >= corr_auto_threshold:
                acao = "CORRIGIR_XML" if cat.startswith("SVA_") else "MANTER_XML"
            elif conf >= suggest_threshold:
                acao = "REVISAR"
            else:
                acao = "REVISAR"
            return pd.Series([cat, conf, motivo, origem, cfop_action, acao])

        df_items[["categoria_fiscal_ia","confianca_ia","motivo_ia","origem_ia","recomendacao_cfop","acao_sugerida"]] = df_items.apply(decide_row, axis=1)

        # Correction ZIP
        out_zip = io.BytesIO()
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
            for x in xml_ativos:
                logical = x["logical_name"]
                base = x["base_name"]
                tree = x["tree"]
                df_dec = df_items.loc[df_items["arquivo"] == logical, ["item","categoria_fiscal_ia","confianca_ia"]].copy()
                corrected = correct_xml_nfcom(tree, df_dec, corr_auto_threshold=corr_auto_threshold, corrigir_descontos=corrigir_descontos)
                base_no_ext = base[:-4] if base.lower().endswith(".xml") else base
                z.writestr(f"{base_no_ext}_processado.xml", corrected)
        out_zip.seek(0)

        # Reports
        total_docs_ativos = len(xml_ativos)
        total_docs_cancelados = len(canceled)
        total_itens = len(df_items)
        total_erros = len(df_errors)
        total_vserv = float(df_items["vServ"].sum()) if "vServ" in df_items.columns else 0.0

        df_cat = (df_items.groupby("categoria_fiscal_ia")
                  .agg(qtd_itens=("arquivo","count"), total_vServ=("vServ","sum"))
                  .reset_index())
        if total_vserv > 0:
            df_cat["participacao_%"] = (df_cat["total_vServ"] / total_vserv) * 100.0
        else:
            df_cat["participacao_%"] = 0.0

        df_cat_file = (df_items.groupby(["arquivo","categoria_fiscal_ia"])
                       .agg(total_vServ=("vServ","sum"), qtd_itens=("arquivo","count"))
                       .reset_index())

        df_revisar = df_items.loc[df_items["confianca_ia"].astype(float) < corr_auto_threshold,
                                  ["arquivo","item","descricao","cClass","CFOP","categoria_fiscal_ia","confianca_ia","motivo_ia","origem_ia","vServ"]].copy()

        if not df_errors.empty and consolidar:
            df_consol = (df_errors.groupby(["regra_id","descricao_regra","mensagem_erro","sugestao_correcao"])
                         .agg(qtd=("arquivo","count"),
                              arquivos=("arquivo", lambda s: ", ".join(sorted(set(s)))))
                         .reset_index())
        else:
            df_consol = pd.DataFrame()

        df_status = pd.DataFrame(
            [{"arquivo_base": x["base_name"], "chave": x.get("chave",""), "status": "ativo"} for x in xml_ativos] +
            canceled
        )

        df_resumo = pd.DataFrame([
            {"Métrica": "Emitente (cliente) CNPJ", "Valor": client_cnpj or ""},
            {"Métrica": "Emitente (cliente) Nome", "Valor": client_nome or ""},
            {"Métrica": "Competência (mês)", "Valor": month_ref or ""},
            {"Métrica": "Docs ativos processados", "Valor": total_docs_ativos},
            {"Métrica": "Docs cancelados descartados", "Valor": total_docs_cancelados},
            {"Métrica": "Total itens", "Valor": total_itens},
            {"Métrica": "Total erros/alertas", "Valor": total_erros},
            {"Métrica": "Total faturado vServ", "Valor": num_to_br(total_vserv)},
            {"Métrica": "Itens p/ revisar (conf < limiar)", "Valor": len(df_revisar)},
        ])

        totals = df_cat.set_index("categoria_fiscal_ia")["total_vServ"].to_dict()
        snapshot = {
            "month": month_ref or datetime.now().strftime("%Y-%m"),
            "emit_cnpj": client_cnpj or "DESCONHECIDO",
            "emit_nome": client_nome or "",
            "total_docs_ativos": total_docs_ativos,
            "total_docs_cancelados": total_docs_cancelados,
            "total_itens": total_itens,
            "total_erros": total_erros,
            "total_vserv": total_vserv,
            "total_scm": float(totals.get("SCM", 0.0)),
            "total_sva_ebook": float(totals.get("SVA_EBOOK", 0.0)),
            "total_sva_locacao": float(totals.get("SVA_LOCACAO", 0.0)),
            "total_sva_tv": float(totals.get("SVA_TV_STREAMING", 0.0)),
            "total_sva_outros": float(totals.get("SVA_OUTROS", 0.0)),
            "itens_revisar": int(len(df_revisar)),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        df_snapshot = pd.DataFrame([snapshot])

        try:
            db_insert_snapshot(snapshot)
        except Exception as e:
            st.warning(f"Histórico não persistiu (ambiente pode ser efêmero). Erro: {e}")

        st.subheader("Resumo do lote")
        st.dataframe(df_resumo, use_container_width=True)

        st.subheader("Totais por categoria fiscal (IA)")
        st.dataframe(df_cat, use_container_width=True)
        st.bar_chart(df_cat.set_index("categoria_fiscal_ia")["total_vServ"])

        st.subheader("Itens para revisar")
        st.dataframe(df_revisar, use_container_width=True)

        st.download_button("Baixar ZIP – XMLs ativos processados", data=out_zip, file_name="xml_nfcom_ativos_processados.zip", mime="application/zip", key="dl_zip")

        # Excel
        try:
            excel_bytes = generate_excel_report(
                df_resumo=df_resumo,
                df_erros=df_errors,
                df_consolidado=df_consol,
                df_detalhe=df_items.drop(columns=["desc_norm"], errors="ignore"),
                df_categoria=df_cat,
                df_categoria_arquivo=df_cat_file,
                df_revisar=df_revisar,
                df_status=df_status,
                df_snapshot=df_snapshot,
            )
            st.download_button(
                "Baixar Excel – Relatório completo",
                data=excel_bytes,
                file_name="relatorio_nfcom_ai.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_xlsx"
            )
        except Exception as e:
            st.error(f"Falha ao gerar Excel (instale openpyxl): {e}")

        # Histórico
        st.markdown("---")
        st.subheader("Histórico por cliente (comparativo mensal)")
        hist = db_load_history(emit_cnpj=client_cnpj) if client_cnpj else db_load_history()
        if hist.empty:
            st.info("Sem histórico ainda.")
        else:
            hist = hist.sort_values(["emit_cnpj","month"])
            st.dataframe(hist, use_container_width=True)
            try:
                pivot = hist.pivot_table(index="month", values=["total_vserv","total_erros"], aggfunc="sum").sort_index()
                st.line_chart(pivot)
            except Exception:
                pass
            st.download_button("Baixar histórico (CSV)", data=hist.to_csv(index=False), file_name="historico_clientes.csv", key="dl_hist")

        st.caption("Nota: Streamlit Cloud pode reiniciar armazenamento. Baixe o histórico periodicamente.")

    st.markdown("<hr><p style='text-align:center;font-size:12px;'>Desenvolvido por Raul Martins – Contare Contabilidade especializada em Provedores de Internet</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
