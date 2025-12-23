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

APP_TITLE = "Validador NFCom 62 + IA (SCM/SVA) – Contare (Aprendizado)"
LOGO_PATH = "Logo-Contare-ISP-1.png"
DB_PATH = os.path.join("data", "history.db")
TRAINING_CSV = os.path.join("data", "training_data.csv")

CATEGORIES = ["SCM", "SVA_EBOOK", "SVA_LOCACAO", "SVA_TV_STREAMING", "SVA_OUTROS"]
AI_ALLOWED = set(CATEGORIES)

st.set_page_config(page_title=APP_TITLE, layout="wide")

# ===============================
# Session State (persistência)
# ===============================
if "results" not in st.session_state:
    st.session_state["results"] = None
if "processed" not in st.session_state:
    st.session_state["processed"] = False


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
# XML helpers
# =========================================================

def parse_xml(file_bytes: bytes) -> etree._ElementTree:
    parser = etree.XMLParser(remove_blank_text=True, recover=True)
    return etree.parse(io.BytesIO(file_bytes), parser)

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
    return first_text(root, ns, ".//n:ide/n:mod | .//ide/mod").strip()

def get_emitente(tree: etree._ElementTree) -> Tuple[str, str]:
    root = tree.getroot()
    ns = get_ns(tree)
    cnpj = first_text(root, ns, ".//n:emit/n:CNPJ | .//emit/CNPJ").strip()
    xnome = first_text(root, ns, ".//n:emit/n:xNome | .//emit/xNome").strip()
    return cnpj, xnome

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
# Cancelamento detection
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
    for n in xp(root, ns, ".//n:xMotivo | .//xMotivo | .//n:xEvento | .//xEvento"):
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
# Training data (aprendizado)
# =========================================================

def training_init():
    os.makedirs(os.path.dirname(TRAINING_CSV), exist_ok=True)
    if not os.path.exists(TRAINING_CSV):
        pd.DataFrame(columns=[
            "emit_cnpj","desc_norm","descricao_exemplo","categoria_aprovada","created_at","source"
        ]).to_csv(TRAINING_CSV, index=False, encoding="utf-8")

@st.cache_data
def training_load() -> pd.DataFrame:
    training_init()
    try:
        return pd.read_csv(TRAINING_CSV, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame(columns=[
            "emit_cnpj","desc_norm","descricao_exemplo","categoria_aprovada","created_at","source"
        ])

def training_lookup_map(df_train: pd.DataFrame, emit_cnpj: str) -> Dict[str, str]:
    m: Dict[str, str] = {}
    if df_train.empty:
        return m

    if emit_cnpj:
        df_c = df_train[df_train["emit_cnpj"] == emit_cnpj]
        for _, r in df_c.iterrows():
            dn, cat = r.get("desc_norm",""), r.get("categoria_aprovada","")
            if dn and cat:
                m[dn] = cat

    df_g = df_train[df_train["emit_cnpj"] == ""]
    for _, r in df_g.iterrows():
        dn, cat = r.get("desc_norm",""), r.get("categoria_aprovada","")
        if dn and cat and dn not in m:
            m[dn] = cat
    return m

def training_append(rows: List[Dict[str, Any]]):
    training_init()
    df = training_load()
    df2 = pd.DataFrame(rows)
    pd.concat([df, df2], ignore_index=True).to_csv(TRAINING_CSV, index=False, encoding="utf-8")
    training_load.clear()

def training_merge_uploaded(uploaded_bytes: bytes):
    training_init()
    try:
        inc = pd.read_csv(io.BytesIO(uploaded_bytes), dtype=str).fillna("")
    except Exception:
        return False, "CSV inválido."
    needed = {"emit_cnpj","desc_norm","categoria_aprovada"}
    if not needed.issubset(set(inc.columns)):
        return False, f"CSV precisa conter colunas: {sorted(list(needed))}"
    df = training_load()
    out = pd.concat([df, inc], ignore_index=True).drop_duplicates(subset=["emit_cnpj","desc_norm"], keep="last")
    out.to_csv(TRAINING_CSV, index=False, encoding="utf-8")
    training_load.clear()
    return True, "Base importada e mesclada com sucesso."


# =========================================================
# Heuristic classifier
# =========================================================

SCM_KEYWORDS = [
    "fibra", "fibra optica", "fibra óptica", "banda larga", "internet", "link",
    "link dedicado", "dedicado", "ftth", "plano", "velocidade", "scm", "dados",
    "conexao", "conexão", "wifi", "wi-fi", "provedor", "acesso", "rede", "conectividade"
]
SVA_EBOOK_KEYWORDS = ["ebook","e-book","livro digital","biblioteca digital","leitura","plataforma de leitura"]
SVA_LOCACAO_KEYWORDS = ["locacao","locação","comodato","aluguel","locar","equipamento","roteador","onu","cpe"]
SVA_TV_KEYWORDS = ["tv","iptv","streaming","conteudo","conteúdo","televisao","televisão"]
SVA_GENERIC_KEYWORDS = ["antivirus","antivírus","backup","email","e-mail","ip fixo","suporte premium","voip","telefonia","sva","cloud","nuvem","seguranca","segurança"]

def heuristic_category(desc: str) -> Tuple[str, float, str]:
    d = normalize_text(desc)
    if not d:
        return ("SVA_OUTROS", 0.50, "Descrição vazia")
    if any(k in d for k in SCM_KEYWORDS) and not any(k in d for k in SVA_GENERIC_KEYWORDS):
        return ("SCM", 0.96, "Palavras-chave fortes de SCM")
    if any(k in d for k in SVA_EBOOK_KEYWORDS) and not any(k in d for k in SCM_KEYWORDS):
        return ("SVA_EBOOK", 0.96, "Palavras-chave eBook")
    if any(k in d for k in SVA_LOCACAO_KEYWORDS) and not any(k in d for k in SCM_KEYWORDS):
        return ("SVA_LOCACAO", 0.95, "Palavras-chave locação/equipamento")
    if any(k in d for k in SVA_TV_KEYWORDS) and not any(k in d for k in SCM_KEYWORDS):
        return ("SVA_TV_STREAMING", 0.95, "Palavras-chave TV/Streaming")
    if any(k in d for k in SVA_GENERIC_KEYWORDS) and not any(k in d for k in SCM_KEYWORDS):
        return ("SVA_OUTROS", 0.90, "Palavras-chave SVA (genérico)")
    if any(k in d for k in SCM_KEYWORDS) and any(k in d for k in SVA_GENERIC_KEYWORDS):
        return ("SCM", 0.70, "Ambíguo (SCM+SVA). Revisar.")
    return ("SVA_OUTROS", 0.60, "Sem evidência forte. IA/Revisão.")


# =========================================================
# OpenAI classifier (robust parsing)
# =========================================================

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
    key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
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
                "origem": "heuristica",
            })
        return out

    payload = [{
        "id": it.get("id"),
        "descricao": (it.get("desc") or "")[:220],
        "cClass": (it.get("cClass") or "")[:32],
        "CFOP": (it.get("cfop") or "")[:16],
    } for it in items]

    user_msg = {"items": payload}

    resp = client.responses.create(
        model=model,
        input=[
            {"role":"system","content":AI_SYSTEM},
            {"role":"user","content":(
                "Classifique os itens abaixo. Retorne SOMENTE JSON válido no formato:\n"
                '{"items":[{"id":"...","categoria_fiscal_ia":"SCM|SVA_EBOOK|SVA_LOCACAO|SVA_TV_STREAMING|SVA_OUTROS","confianca_ia":0.0-1.0,"motivo_ia":"..."}]}\n\n'
                + json.dumps(user_msg, ensure_ascii=False)
            )},
        ],
        temperature=0.0,
        max_output_tokens=1400,
    )

    raw_text = (resp.output_text or "").strip()

    def _strip_fences(t: str) -> str:
        t = (t or "").strip()
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
        t = re.sub(r"\s*```$", "", t)
        return t.strip()

    def _extract_json_object(t: str) -> Optional[str]:
        t = _strip_fences(t)
        start = t.find("{")
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return t[start:i+1]
        return None

    def _safe_json_loads(t: str) -> Optional[dict]:
        t = _strip_fences(t)
        try:
            obj = json.loads(t)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        js = _extract_json_object(t)
        if not js:
            return None
        js_fixed = js.replace("\u201c", '"').replace("\u201d", '"')
        js_fixed = re.sub(r",\s*([}\]])", r"\1", js_fixed)
        try:
            obj = json.loads(js_fixed)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
        return None

    data = _safe_json_loads(raw_text)
    if not data or "items" not in data:
        out = []
        for it in items:
            cat, conf, why = heuristic_category(it.get("desc", ""))
            out.append({
                "id": it.get("id"),
                "categoria_fiscal_ia": cat,
                "confianca_ia": float(conf),
                "motivo_ia": f"Resposta IA inválida (parser). Heurística: {why}",
                "origem": "fallback_parser",
            })
        return out

    results = data.get("items", [])
    by_id = {r.get("id"): r for r in results if isinstance(r, dict)}

    out = []
    for it in items:
        r = by_id.get(it.get("id"))
        if not r:
            cat, conf, why = heuristic_category(it.get("desc", ""))
            out.append({
                "id": it.get("id"),
                "categoria_fiscal_ia": cat,
                "confianca_ia": float(conf),
                "motivo_ia": f"IA sem item correspondente. Heurística: {why}",
                "origem": "fallback_sem_item",
            })
            continue

        cat = (r.get("categoria_fiscal_ia") or "SVA_OUTROS").strip()
        if cat not in AI_ALLOWED:
            cat = "SVA_OUTROS"

        try:
            conf = float(r.get("confianca_ia", 0.6) or 0.6)
        except Exception:
            conf = 0.6
        conf = max(0.0, min(1.0, conf))
        motivo = (r.get("motivo_ia") or "").strip()[:220]

        out.append({
            "id": it.get("id"),
            "categoria_fiscal_ia": cat,
            "confianca_ia": conf,
            "motivo_ia": motivo,
            "origem": "openai",
        })
    return out


# =========================================================
# Items extraction + correction + per-XML change log
# =========================================================

def extract_items_nfcom(tree: etree._ElementTree, file_name: str) -> List[Dict[str, Any]]:
    root = tree.getroot()
    ns = get_ns(tree)
    dets = xp(root, ns, ".//n:det | .//det")
    items = []
    for idx, det in enumerate(dets, start=1):
        cclass = first_text(det, ns, "./n:prod/n:cClass | ./prod/cClass")
        xprod  = first_text(det, ns, "./n:prod/n:xProd | ./prod/xProd")
        cfop   = first_text(det, ns, "./n:prod/n:CFOP | ./prod/CFOP")
        vitem  = to_float(first_text(det, ns, "./n:prod/n:vItem | ./prod/vItem"))
        vprod  = to_float(first_text(det, ns, "./n:prod/n:vProd | ./prod/vProd"))
        vdesc  = to_float(first_text(det, ns, "./n:prod/n:vDesc | ./prod/vDesc"))
        vout   = to_float(first_text(det, ns, "./n:prod/n:vOutro | ./prod/vOutro"))
        items.append({
            "arquivo": file_name,
            "item": idx,
            "cClass": cclass,
            "descricao": xprod,
            "CFOP": cfop,
            "vItem": float(vitem),
            "vProd": float(vprod),
            "vDesc": float(vdesc),
            "vOutros": float(vout),
            "vServ": float(vprod),
        })
    return items

def simulate_and_or_correct_xml_nfcom(
    tree: etree._ElementTree,
    df_dec: pd.DataFrame,
    corr_auto_threshold: float,
    corrigir_descontos: bool,
    apply_changes: bool,
) -> Tuple[bytes, List[Dict[str, Any]], bool]:
    """
    Retorna:
      - xml_saida (original se apply_changes=False; corrigido se True)
      - changes (lista detalhada do que mudaria/mudou)
      - changed_flag (True se existe alguma mudança)
    """
    root = tree.getroot()
    original_xml = etree.tostring(tree, encoding="utf-8", xml_declaration=True)

    copy_root = etree.fromstring(etree.tostring(root))
    new_tree = etree.ElementTree(copy_root)
    ns = get_ns(new_tree)

    decisions = {int(r["item"]): (str(r["categoria_fiscal_ia"]), float(r["confianca_ia"])) for _, r in df_dec.iterrows()}

    dets = xp(copy_root, ns, ".//n:det | .//det")
    changes: List[Dict[str, Any]] = []

    for idx, det in enumerate(dets, start=1):
        cat, conf = decisions.get(idx, ("SVA_OUTROS", 0.0))

        # 1) Remover CFOP de SVA com confiança >= threshold
        if cat.startswith("SVA_") and conf >= corr_auto_threshold:
            cfop_nodes = xp(det, ns, "./n:prod/n:CFOP | ./prod/CFOP")
            if cfop_nodes:
                old_cfop = (cfop_nodes[0].text or "").strip()
                changes.append({
                    "item": idx,
                    "acao": "REMOVER_CFOP_SVA",
                    "detalhe": f"Remover CFOP='{old_cfop}' (cat={cat}, conf={conf:.2f})",
                })
                if apply_changes:
                    for node in cfop_nodes:
                        parent = node.getparent()
                        if parent is not None:
                            parent.remove(node)

        # 2) Paliativo desconto (vProd = vItem quando vProd < vItem)
        if corrigir_descontos:
            vitem_nodes = xp(det, ns, "./n:prod/n:vItem | ./prod/vItem")
            vprod_nodes = xp(det, ns, "./n:prod/n:vProd | ./prod/vProd")
            if vitem_nodes and vprod_nodes:
                vi_text = (vitem_nodes[0].text or "").strip()
                vp_text = (vprod_nodes[0].text or "").strip()
                vi = to_float(vi_text)
                vp = to_float(vp_text)
                if vp < vi:
                    changes.append({
                        "item": idx,
                        "acao": "AJUSTAR_VPROD",
                        "detalhe": f"vProd {vp_text} -> {vi_text} (paliativo desconto)",
                    })
                    if apply_changes:
                        vprod_nodes[0].text = vi_text

    changed_flag = len(changes) > 0
    if apply_changes:
        new_xml = etree.tostring(new_tree, encoding="utf-8", xml_declaration=True)
        return new_xml, changes, changed_flag

    return original_xml, changes, changed_flag


# =========================================================
# Excel report (openpyxl)
# =========================================================

def generate_excel_report(**dfs) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet, df in dfs.items():
            if df is None:
                continue
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(writer, sheet_name=str(sheet)[:31], index=False)
    output.seek(0)
    return output.read()


# =========================================================
# Main UI
# =========================================================

def main():
    training_init()

    c1, c2 = st.columns([1, 4])
    with c1:
        try:
            st.image(LOGO_PATH)
        except Exception:
            st.write("")
    with c2:
        st.markdown(f"## {APP_TITLE}")
        st.caption("Desenvolvido por Raul Martins — Contare Contabilidade especializada em Provedores de Internet")

    st.sidebar.header("Configurações")

    # Toggle: aplicar correção agora vs apenas sugerir
    apply_changes = st.sidebar.radio(
        "Correção dos XMLs",
        options=["Aplicar correções no XML", "Apenas sugerir (não altera XML)"],
        index=0,
        help="Se 'Apenas sugerir', o ZIP contém os XMLs originais, mas com logs de mudanças sugeridas."
    ) == "Aplicar correções no XML"

    corrigir_descontos = st.sidebar.checkbox("Paliativo descontos: vProd = vItem quando vProd < vItem", value=False)

    enable_ai = st.sidebar.checkbox("Ativar IA (OpenAI) para classificação", value=True)
    ai_model = st.sidebar.text_input("Modelo OpenAI", value="gpt-4o-mini")

    corr_auto_threshold = st.sidebar.slider("Limiar para remover CFOP do SVA (confiança mínima)", 0.50, 1.00, 0.95, 0.01)
    suggest_threshold = st.sidebar.slider("Limiar para sugestão forte (sem IA)", 0.50, 1.00, 0.85, 0.01)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Base de aprendizado (SCM/SVA)")

    up_train = st.sidebar.file_uploader("Importar base (CSV)", type=["csv"])
    if up_train is not None:
        ok, msg = training_merge_uploaded(up_train.read())
        st.sidebar.success(msg) if ok else st.sidebar.error(msg)

    df_train = training_load()
    st.sidebar.download_button("Baixar base (CSV)", data=df_train.to_csv(index=False).encode("utf-8"), file_name="training_data.csv", key="dl_training_csv")

    st.sidebar.markdown("---")
    cancel_file = st.sidebar.file_uploader("Chaves canceladas (CSV/TXT)", type=["csv", "txt"])
    cancel_keys = set()
    if cancel_file is not None:
        raw = cancel_file.read()
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            text = raw.decode("latin1", errors="ignore")
        cancel_keys = set(re.findall(r"\d{44}", text))

    uploaded = st.file_uploader("Envie XML (NFCom) ou ZIP com XMLs", type=["xml", "zip"], accept_multiple_files=True)

    # =======================
    # PROCESSAMENTO DO LOTE
    # =======================
    process_btn = st.button("Processar lote", type="primary", disabled=not bool(uploaded))
    if process_btn and uploaded:
        items_all: List[Dict[str, Any]] = []
        xml_ativos: List[Dict[str, Any]] = []
        canceled: List[Dict[str, Any]] = []
        invalid: List[Dict[str, Any]] = []

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

            try:
                tree = parse_xml(xml_bytes)
            except Exception as e:
                invalid.append({"arquivo": logical_name, "erro": f"XML inválido: {e}"})
                return

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

            items_all.extend(extract_items_nfcom(tree, logical_name))
            xml_ativos.append({"base_name": base_name, "logical_name": logical_name, "tree": tree, "chave": chave})

        for f in uploaded:
            fname = f.name
            content = f.read()
            if fname.lower().endswith(".zip"):
                try:
                    with zipfile.ZipFile(io.BytesIO(content)) as zf:
                        for info in zf.infolist():
                            if info.filename.lower().endswith(".xml"):
                                base_name = info.filename.replace("\\", "/").replace("/", "_")
                                handle_xml(zf.read(info), base_name, f"{fname}::{info.filename}")
                except zipfile.BadZipFile:
                    invalid.append({"arquivo": fname, "erro": "ZIP inválido/corrompido."})
            else:
                handle_xml(content, fname, fname)

        df_items = pd.DataFrame(items_all) if items_all else pd.DataFrame()
        if df_items.empty:
            st.session_state["results"] = {"empty": True, "invalid": invalid, "canceled": canceled}
            st.session_state["processed"] = True
            st.rerun()

        # ========= classificação (aprendizado > heurística > IA) =========
        df_items["desc_norm"] = df_items["descricao"].fillna("").map(normalize_text)
        df_items["id_desc"] = df_items["desc_norm"].map(lambda x: re.sub(r"[^a-z0-9]+", "_", x)[:80])

        train_map = training_lookup_map(df_train, client_cnpj or "")
        df_items["categoria_training"] = df_items["desc_norm"].map(lambda dn: train_map.get(dn, ""))
        df_items["confianca_training"] = df_items["categoria_training"].map(lambda c: 1.0 if c in AI_ALLOWED else 0.0)

        heur = df_items["descricao"].fillna("").map(lambda d: heuristic_category(d))
        df_items["categoria_heur"], df_items["confianca_heur"], df_items["motivo_heur"] = zip(*heur)

        need_ai_mask = (
            enable_ai
            & (df_items["confianca_training"].astype(float) < 0.99)
            & (df_items["confianca_heur"].astype(float) < suggest_threshold)
        )
        df_need = df_items.loc[need_ai_mask, ["id_desc","descricao","cClass","CFOP"]].drop_duplicates("id_desc").copy()

        ai_by_id: Dict[str, Dict[str, Any]] = {}
        if enable_ai and not df_need.empty:
            batch = [{"id": r["id_desc"], "desc": r["descricao"], "cClass": r.get("cClass",""), "cfop": r.get("CFOP","")} for _, r in df_need.iterrows()]
            for i in range(0, len(batch), 50):
                chunk = batch[i:i+50]
                for r in ai_classify_batch(chunk, model=ai_model):
                    ai_by_id[r["id"]] = r

        def decide_row(r):
            if r.get("categoria_training","") in AI_ALLOWED:
                cat = r["categoria_training"]; conf = 1.0
                motivo = "Aprovação do escritório (base de aprendizado)"; origem = "aprendizado"
            else:
                cat = r["categoria_heur"]; conf = float(r["confianca_heur"])
                motivo = r["motivo_heur"]; origem = "heuristica"
                ai = ai_by_id.get(r["id_desc"])
                if ai is not None and conf < 0.95:
                    cat = ai.get("categoria_fiscal_ia", cat)
                    conf = float(ai.get("confianca_ia", conf))
                    motivo = ai.get("motivo_ia", motivo)
                    origem = ai.get("origem", "openai")
            if cat not in AI_ALLOWED:
                cat = "SVA_OUTROS"
            return pd.Series([cat, conf, motivo, origem])

        df_items[["categoria_fiscal_ia","confianca_ia","motivo_ia","origem_ia"]] = df_items.apply(decide_row, axis=1)

        # ========= fila de revisão =========
        df_revisar = df_items.loc[
            (df_items["origem_ia"] != "aprendizado") & (df_items["confianca_ia"].astype(float) < corr_auto_threshold),
            ["arquivo","item","descricao","cClass","CFOP","categoria_fiscal_ia","confianca_ia","motivo_ia","origem_ia","vServ","desc_norm"]
        ].copy()

        # ========= pré-gerar logs e XMLs para preview e downloads =========
        per_file = []
        changes_all = []
        for x in xml_ativos:
            logical = x["logical_name"]
            base = x["base_name"]
            tree = x["tree"]

            df_dec = df_items.loc[df_items["arquivo"] == logical, ["item","categoria_fiscal_ia","confianca_ia"]].copy()

            xml_out, changes, changed_flag = simulate_and_or_correct_xml_nfcom(
                tree,
                df_dec=df_dec,
                corr_auto_threshold=corr_auto_threshold,
                corrigir_descontos=corrigir_descontos,
                apply_changes=apply_changes,
            )

            orig_bytes = etree.tostring(tree, encoding="utf-8", xml_declaration=True)

            per_file.append({
                "arquivo": logical,
                "base_name": base,
                "chave": x.get("chave",""),
                "changed": bool(changed_flag),
                "changes_count": len(changes),
                "xml_original": orig_bytes,
                "xml_saida": xml_out,
            })

            for c in changes:
                changes_all.append({
                    "arquivo": logical,
                    "base_name": base,
                    "chave": x.get("chave",""),
                    "item": c.get("item"),
                    "acao": c.get("acao"),
                    "detalhe": c.get("detalhe"),
                    "modo": "APLICADO" if apply_changes else "SUGERIDO",
                })

        df_files = pd.DataFrame(per_file)
        df_changes = pd.DataFrame(changes_all) if changes_all else pd.DataFrame(columns=["arquivo","base_name","chave","item","acao","detalhe","modo"])

        # ========= persistência no estado =========
        st.session_state["results"] = {
            "empty": False,
            "client_cnpj": client_cnpj,
            "client_nome": client_nome,
            "month_ref": month_ref,
            "df_items": df_items,
            "df_revisar": df_revisar,
            "df_files": df_files,
            "df_changes": df_changes,
            "invalid": invalid,
            "canceled": canceled,
            "apply_changes": apply_changes,
        }
        st.session_state["processed"] = True
        st.rerun()

    # =======================
    # RENDER (state)
    # =======================
    if not st.session_state.get("processed") or not st.session_state.get("results"):
        st.info("Envie os arquivos e clique em **Processar lote**.")
        return

    res = st.session_state["results"]
    if res.get("empty"):
        st.warning("Nenhum XML NFCom ativo foi processado.")
        if res.get("invalid"):
            st.subheader("Ignorados")
            st.dataframe(pd.DataFrame(res["invalid"]), use_container_width=True)
        if res.get("canceled"):
            st.subheader("Cancelados/Descartados")
            st.dataframe(pd.DataFrame(res["canceled"]), use_container_width=True)
        return

    df_items = res["df_items"]
    df_revisar = res["df_revisar"]
    df_files = res["df_files"]
    df_changes = res["df_changes"]
    client_cnpj = res.get("client_cnpj","")
    client_nome = res.get("client_nome","")
    month_ref = res.get("month_ref","")
    apply_changes = bool(res.get("apply_changes", True))

    # ========= revisão =========
    st.subheader("Fila de revisão (aprove e o sistema aprende)")
    if df_revisar.empty:
        st.success("Nenhum item precisa de revisão neste lote.")
    else:
        editor_df = df_revisar[["arquivo","item","descricao","cClass","categoria_fiscal_ia","confianca_ia","origem_ia","vServ"]].copy()
        editor_df["categoria_aprovada"] = editor_df["categoria_fiscal_ia"]
        edited = st.data_editor(
            editor_df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={"categoria_aprovada": st.column_config.SelectboxColumn("categoria_aprovada", options=CATEGORIES, required=True)},
            key="editor_revisao",
        )
        if st.button("Salvar aprovações na base"):
            merged = df_revisar.merge(edited[["arquivo","item","categoria_aprovada"]], on=["arquivo","item"], how="left")
            now = datetime.now().isoformat(timespec="seconds")
            rows = []
            for _, rr in merged.iterrows():
                cat = rr.get("categoria_aprovada","")
                if cat in AI_ALLOWED and rr.get("desc_norm",""):
                    rows.append({
                        "emit_cnpj": client_cnpj or "",
                        "desc_norm": rr["desc_norm"],
                        "descricao_exemplo": (rr.get("descricao","") or "")[:250],
                        "categoria_aprovada": cat,
                        "created_at": now,
                        "source": "aprovacao_app",
                    })
            if rows:
                training_append(rows)
                st.success(f"Aprovações salvas: {len(rows)}. Reprocessar aplicará 100% de confiança nessas descrições.")
            else:
                st.warning("Nenhuma aprovação válida selecionada.")

    # ========= dashboard =========
    st.subheader("Dashboard do lote")
    total_docs = int(len(df_files))
    total_changed = int(df_files["changed"].sum()) if "changed" in df_files.columns else 0
    total_itens = int(len(df_items))
    total_vserv = float(df_items["vServ"].sum()) if "vServ" in df_items.columns else 0.0
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Docs ativos", total_docs)
    k2.metric("Docs com mudança", total_changed)
    k3.metric("Total itens", total_itens)
    k4.metric("Total vServ", num_to_br(total_vserv))

    df_cat = df_items.groupby("categoria_fiscal_ia").agg(qtd_itens=("arquivo","count"), total_vServ=("vServ","sum")).reset_index()
    st.dataframe(df_cat, use_container_width=True)
    st.bar_chart(df_cat.set_index("categoria_fiscal_ia")["total_vServ"])

    # ========= log =========
    st.subheader("Log de mudanças por XML")
    if df_changes.empty:
        st.info("Nenhuma mudança proposta/aplicada neste lote.")
    else:
        st.dataframe(df_changes, use_container_width=True)

    # ========= preview =========
    st.subheader("Pré-visualização (antes de baixar)")
    st.caption("Selecione um XML e veja: mudanças propostas/aplicadas + XML original vs XML de saída.")

    opts = df_files["arquivo"].tolist()
    sel = st.selectbox("Escolha o arquivo", options=opts, key="preview_file")
    row = df_files[df_files["arquivo"] == sel].iloc[0]

    colA, colB = st.columns([1, 2])
    with colA:
        st.write("**Resumo**")
        st.write(f"- Base: `{row['base_name']}`")
        st.write(f"- Chave: `{row.get('chave','')}`")
        st.write(f"- Mudanças: **{int(row.get('changes_count',0))}**")
        st.write(f"- Modo: **{'Aplicado' if apply_changes else 'Apenas sugerido'}**")
        df_c = df_changes[df_changes["arquivo"] == sel].copy() if not df_changes.empty else pd.DataFrame()
        if df_c.empty:
            st.success("Nenhuma mudança para este XML.")
        else:
            st.write("**Mudanças**")
            st.dataframe(df_c[["item","acao","detalhe","modo"]], use_container_width=True, height=260)

    with colB:
        tab1, tab2 = st.tabs(["XML original", "XML de saída"])
        with tab1:
            st.text_area("original", value=(row["xml_original"].decode("utf-8", errors="ignore")), height=420)
        with tab2:
            st.text_area("saida", value=(row["xml_saida"].decode("utf-8", errors="ignore")), height=420)

    # ========= downloads =========
    st.subheader("Downloads")

    def build_zip(filter_mode: str) -> bytes:
        """
        filter_mode:
          - 'todos'
          - 'somente_corrigidos' (changed=True)
          - 'somente_sem_mudanca' (changed=False)
        Sempre inclui: status_por_xml.csv e log_mudancas.csv
        """
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("status_por_xml.csv", df_files[["arquivo","base_name","chave","changed","changes_count"]].to_csv(index=False))
            z.writestr("log_mudancas.csv", df_changes.to_csv(index=False) if not df_changes.empty else "arquivo,base_name,chave,item,acao,detalhe,modo\n")

            for _, r in df_files.iterrows():
                if filter_mode == "somente_corrigidos" and not bool(r["changed"]):
                    continue
                if filter_mode == "somente_sem_mudanca" and bool(r["changed"]):
                    continue

                base = r["base_name"]
                base_no_ext = base[:-4] if base.lower().endswith(".xml") else base
                sufixo = "_corrigido" if bool(r["changed"]) and apply_changes else ("_sugerido" if bool(r["changed"]) else "_validado")
                z.writestr(f"{base_no_ext}{sufixo}.xml", r["xml_saida"])
        out.seek(0)
        return out.read()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("Baixar ZIP – Todos ativos", data=build_zip("todos"), file_name="xml_ativos.zip", mime="application/zip", key="dl_zip_todos")
    with col2:
        st.download_button("Baixar ZIP – Somente com mudança", data=build_zip("somente_corrigidos"), file_name="xml_somente_mudanca.zip", mime="application/zip", key="dl_zip_mudanca")
    with col3:
        st.download_button("Baixar ZIP – Somente sem mudança", data=build_zip("somente_sem_mudanca"), file_name="xml_sem_mudanca.zip", mime="application/zip", key="dl_zip_sem_mudanca")

    st.markdown("---")
    st.subheader("Relatório (Excel)")
    try:
        excel = generate_excel_report(
            Resumo=pd.DataFrame([{"cliente_cnpj": client_cnpj, "cliente_nome": client_nome, "competencia": month_ref, "modo": "APLICADO" if apply_changes else "SUGERIDO"}]),
            Detalhamento_Itens=df_items.drop(columns=["desc_norm"], errors="ignore"),
            Resumo_Categoria=df_cat,
            Arquivos_Status=df_files[["arquivo","base_name","chave","changed","changes_count"]],
            Log_Mudancas=df_changes,
            Itens_Revisao=df_revisar.drop(columns=["desc_norm"], errors="ignore"),
            Base_Aprendizado=training_load(),
        )
        st.download_button(
            "Baixar Excel – Relatório completo",
            data=excel,
            file_name="relatorio_nfcom_aprendizado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_excel",
        )
    except Exception as e:
        st.warning(f"Não foi possível gerar Excel. Verifique se 'openpyxl' está instalado. Erro: {e}")

    st.markdown("<hr><p style='text-align:center;font-size:12px;'>Desenvolvido por Raul Martins – Contare</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
