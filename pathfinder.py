import os
import time
import json
import secrets
import pathlib
import datetime
import requests
import urllib3
from flask import (
    Flask, request, redirect, url_for, render_template,
    session, jsonify, abort
)
from dotenv import load_dotenv
from collections import deque, OrderedDict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -------------------------------------------------------------
# Config
# -------------------------------------------------------------
load_dotenv()
DEBUG_FS = os.getenv("FLASK_DEBUG", "0") == "1"

CLIENT_ID     = os.getenv("FAMILYSEARCH_APP_KEY") or ""
REDIRECT_URI  = os.getenv("FAMILYSEARCH_REDIRECT_URI", "https://127.0.0.1:5000/callback")
AUTH_URL      = "https://identbeta.familysearch.org/cis-web/oauth2/v3/authorization"
TOKEN_URL     = "https://identbeta.familysearch.org/cis-web/oauth2/v3/token"
API_BASE_URL  = "https://apibeta.familysearch.org"
SCOPE         = "openid profile email"

# Diretório onde ficam os snapshots públicos (somente leitura)
BASE_DIR  = pathlib.Path(__file__).resolve().parent
SHARE_DIR = BASE_DIR / "shares"
SHARE_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
# Segurança de sessão
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24))
app.config.update(
    SESSION_COOKIE_SECURE=True,         # exige https em produção
    SESSION_COOKIE_HTTPONLY=True,       # JS não acessa cookies
    SESSION_COOKIE_SAMESITE="Lax",      # segura contra CSRF básico
)

# HTTP session with retries
session_http = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=0.3,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=False,
)
adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=retries)
session_http.mount("https://", adapter)
session_http.mount("http://", adapter)
DEFAULT_TIMEOUT = 8

# -------------------------------------------------------------
# Rate limit MUITO simples por IP (janela de 60s)
# -------------------------------------------------------------
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))
_rate_window = 60
_rate_hits = {}  # ip -> deque[timestamps]

@app.before_request
def _apply_rate_limit():
    # rotas estáticas e páginas de leitura não batem em API
    path = request.path
    if path.startswith("/static") or path.startswith("/share"):
        return
    if request.method == "GET" and path in ("/", "/login", "/callback", "/about", "/view"):
        # ainda assim protegemos /view pois chama RF
        pass

    ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request.remote_addr or "0.0.0.0"
    now = time.time()
    dq = _rate_hits.setdefault(ip, deque())
    # purge janelas antigas
    while dq and now - dq[0] > _rate_window:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_PER_MIN:
        abort(429)
    dq.append(now)


@app.errorhandler(429)
def _too_many(e):
    msg = "Muitas solicitações em pouco tempo. Aguarde um instante e tente novamente."
    # Para chamadas API:
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": msg}), 429
    return render_template("search.html", error=msg, max_depth=8), 429


# -------------------------------------------------------------
# Helpers OAuth / Tokens
# -------------------------------------------------------------
def build_auth_url():
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPE,
    }
    q = "&".join(f"{k}={requests.utils.quote(v)}" for k, v in params.items())
    return f"{AUTH_URL}?{q}"

def exchange_code_for_token(code: str):
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": CLIENT_ID,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    r = session_http.post(TOKEN_URL, data=data, headers=headers, timeout=DEFAULT_TIMEOUT)
    if DEBUG_FS:
        try:
            print("[OAuth] token resp", r.status_code, r.json())
        except Exception:
            print("[OAuth] token resp", r.status_code, r.text[:200])
    if r.status_code != 200:
        return None
    return r.json()

def _client_ip_fallback():
    if request and request.headers.get("X-Forwarded-For"):
        return request.headers["X-Forwarded-For"].split(",")[0].strip()
    if request and request.remote_addr:
        return request.remote_addr
    return "127.0.0.1"

def get_unauth_token():
    data = {
        "grant_type": "unauthenticated_session",
        "client_id": CLIENT_ID,
        "ip_address": _client_ip_fallback(),  # exigido no beta/integ
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    try:
        r = session_http.post(TOKEN_URL, data=data, headers=headers, timeout=DEFAULT_TIMEOUT)
        if DEBUG_FS:
            try:
                print("[OAuth] unauth resp", r.status_code, r.json())
            except Exception:
                print("[OAuth] unauth resp", r.status_code, r.text[:200])
        if r.status_code == 200:
            j = r.json()
            return j.get("access_token")
    except requests.exceptions.RequestException as e:
        if DEBUG_FS: print("[OAuth] unauth EXC:", e)
    return None

def get_headers(access_token=None):
    token = access_token or session.get("access_token")
    if not token:
        return None
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "PathFinder/2.0",
    }

# -------------------------------------------------------------
# Cache TTL para /persons/{id}
# -------------------------------------------------------------
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "900"))  # 15 min padrão
CACHE_MAX = int(os.getenv("CACHE_MAX", "2500"))

class TTLCache:
    def __init__(self, ttl=CACHE_TTL, max_items=CACHE_MAX):
        self.ttl = ttl
        self.max = max_items
        self.data = OrderedDict()  # pid -> (ts, value)

    def get(self, key):
        if key in self.data:
            ts, val = self.data[key]
            if time.time() - ts < self.ttl:
                # move p/ fim (recente)
                self.data.move_to_end(key)
                return val
            else:
                del self.data[key]
        return None

    def set(self, key, value):
        self.data[key] = (time.time(), value)
        self.data.move_to_end(key)
        # evict
        if len(self.data) > self.max:
            self.data.popitem(last=False)

_person_cache = TTLCache()

# -------------------------------------------------------------
# FamilySearch API helpers (com cache)
# -------------------------------------------------------------
# (Substitua a sua função get_person_with_relatives por esta)
def get_person_with_relatives(person_id, headers):
    """
    Lê /platform/tree/persons/{id} (JSON) e extrai:
      - details
      - parents
      - children
      - spouses
    Usa cache TTL simples.
    """
    cached = _person_cache.get(person_id)
    if cached is not None:
        if DEBUG_FS: print(f"[DEBUG][Cache] Cache HIT para {person_id}") # >>> NOVO DEBUG
        return cached

    url = f"{API_BASE_URL}/platform/tree/persons/{person_id}"
    
    # >>> NOVO DEBUG: Log antes de fazer a chamada
    if DEBUG_FS: print(f"[DEBUG] Buscando na API para: {person_id} | URL: {url}")
    
    try:
        r = session_http.get(url, headers=headers, verify=False, timeout=DEFAULT_TIMEOUT)
        
        # >>> NOVO DEBUG: Log do status da resposta
        if DEBUG_FS: print(f"[DEBUG] Resposta da API para {person_id}. Status: {r.status_code}")
        
    except requests.exceptions.RequestException as e:
        # >>> NOVO DEBUG: Log de erro de conexão
        if DEBUG_FS: print(f"[DEBUG] ERRO DE CONEXÃO para {person_id}: {e}")
        result = (None, [], [], [])
        _person_cache.set(person_id, result)
        return result

    parents, children, spouses = set(), set(), set()
    details = None

    if r.status_code == 200:
        data = r.json()
        
        if data.get("persons"):
            details = data["persons"][0]

        # pais/filhos
        for rel in data.get("childAndParentsRelationships", []) or []:
            child_id = (rel.get("child") or {}).get("resourceId")
            p1 = (rel.get("parent1") or {}).get("resourceId")
            p2 = (rel.get("parent2") or {}).get("resourceId")
            if child_id == person_id:
                if p1: parents.add(p1)
                if p2: parents.add(p2)
            if p1 == person_id or p2 == person_id:
                if child_id: children.add(child_id)

        # cônjuges
        for rel in data.get("relationships", []) or []:
            if rel.get("type") == "http://gedcomx.org/Couple":
                a = (rel.get("person1") or {}).get("resourceId")
                b = (rel.get("person2") or {}).get("resourceId")
                if a == person_id and b: spouses.add(b)
                elif b == person_id and a: spouses.add(a)
        
        # >>> NOVO DEBUG: Log dos dados extraídos
        if DEBUG_FS: print(f"[DEBUG] Dados extraídos para {person_id}: Pais={list(parents)}, Cônjuges={list(spouses)}")

    else:
        # >>> NOVO DEBUG: Log se a resposta da API não for 200 OK
        if DEBUG_FS: print(f"[DEBUG] Resposta de erro da API para {person_id}: {r.text[:500]}")


    result = (details, list(parents), list(children), list(spouses))
    _person_cache.set(person_id, result)
    return result

def get_person_name(pid, headers):
    det, _, _, _ = get_person_with_relatives(pid, headers)
    if not det:
        return pid
    disp = det.get("display") or {}
    return disp.get("name") or det.get("id") or pid

# -------------------------------------------------------------
# Variantes & pós-processamento (mesmo do seu app)
# -------------------------------------------------------------
K_PATHS_PER_NODE = 16

def _edge_sig(path):
    L = len(path)
    if L >= 3: return (path[-3], path[-2], path[-1])
    if L == 2: return (path[-2], path[-1])
    return (path[-1],)

def _add_path_variant(visited, node, new_path, k=K_PATHS_PER_NODE):
    cur = visited.get(node)
    if not cur:
        visited[node] = [new_path]
        return True
    sig = _edge_sig(new_path)
    if any(_edge_sig(p) == sig for p in cur):
        return False
    if len(cur) < k:
        cur.append(new_path)
        return True
    return False

def post_process_paths(paths_with_ancestors, headers, keep_within=3, max_paths=8):
    if not paths_with_ancestors:
        return []
    # 1) remove loops e duplica por conjunto de nós
    valid, seen_sets = [], set()
    for path, ancestor in paths_with_ancestors:
        if len(set(path)) != len(path):
            continue
        sig = frozenset(path)
        if sig not in seen_sets:
            valid.append((path[:], ancestor))
            seen_sets.add(sig)
    if not valid:
        return []

    # 2) consolida casais
    consolidated, used = [], set()
    for i in range(len(valid)):
        if i in used: continue
        p1, a1 = valid[i]; match = False
        for j in range(i + 1, len(valid)):
            if j in used: continue
            p2, a2 = valid[j]
            if len(p1) != len(p2): continue
            diff_idx, diff_count = -1, 0
            for k in range(len(p1)):
                if p1[k] != p2[k]:
                    diff_idx = k; diff_count += 1
            if diff_count == 1:
                n1, n2 = p1[diff_idx], p2[diff_idx]
                _, _, _, sp1 = get_person_with_relatives(n1, headers)
                if n2 in sp1:
                    merged_anc = tuple(sorted((n1, n2)))
                    merged_path = p1[:diff_idx] + [merged_anc] + p1[diff_idx + 1:]
                    consolidated.append((merged_path, merged_anc))
                    used.add(i); used.add(j); match = True; break
        if not match:
            consolidated.append((p1, a1))
    if not consolidated:
        return []

    # 3) ordenar por tamanho, manter até (min + keep_within)
    dedup_order, seen_order = [], set()
    for p, a in consolidated:
        t = tuple(p)
        if t in seen_order: continue
        seen_order.add(t); dedup_order.append((p, a))
    dedup_order.sort(key=lambda x: len(x[0]))

    min_len = len(dedup_order[0][0])
    candidates = [(p, a) for (p, a) in dedup_order if len(p) <= min_len + keep_within]

    # 4) diversidade
    selected, seen_variant = [], set()

    def couple_child_variant(path):
        for idx, node in enumerate(path):
            if isinstance(node, tuple):
                if idx + 1 < len(path):
                    child = path[idx + 1]
                    if not isinstance(child, tuple):
                        return (tuple(sorted(node)), child)
                return (tuple(sorted(node)), None)
        return (None, None)

    for p, a in candidates:
        var = couple_child_variant(p)
        if var in seen_variant: continue
        seen_variant.add(var); selected.append((p, a))
        if len(selected) >= max_paths: break

    if len(selected) <= 1 and len(candidates) > 1:
        selected = candidates[:max_paths]
    return selected

# -------------------------------------------------------------
# Busca (BFS) – sobe APENAS por pais dos dois lados
# -------------------------------------------------------------
# (Substitua a sua função find_paths por esta)
# (Substitua a sua função find_paths por esta versão corrigida)
def find_paths(person1_id, person2_id, headers, max_depth=8):
    max_nodes = 10000
    expanded = 0

    q1, q2 = deque(), deque()
    visited1, visited2 = {}, {}

    q1.append((person1_id, [person1_id]))
    visited1[person1_id] = [[person1_id]]

    q2.append((person2_id, [person2_id]))
    visited2[person2_id] = [[person2_id]]

    paths_with_ancestors = []
    depth = 0
    best_len = None

    # O loop principal continua enquanto houver nós para expandir e os limites não forem atingidos
    while depth < max_depth and (q1 or q2) and len(paths_with_ancestors) < 50 and expanded < max_nodes:
        
        if DEBUG_FS: print(f"\n[DEBUG] Profundidade: {depth}, Fila1: {len(q1)}, Fila2: {len(q2)}, Nós Expandidos: {expanded}")
        
        # >>> INÍCIO DA CORREÇÃO <<<
        # Processa a fronteira 1 apenas se a fila não estiver vazia
        if q1:
            q1_size = len(q1)
            for _ in range(q1_size):
                curr_id, path = q1.popleft()
                if DEBUG_FS: print(f"[DEBUG][Lado 1] Processando: {curr_id}")

                if curr_id in visited2:
                    if DEBUG_FS: print(f"[DEBUG][ENCONTRO!] ID {curr_id} achado em ambas as buscas.")
                    for p2 in visited2[curr_id]:
                        full_path = path + p2[::-1][1:]
                        paths_with_ancestors.append((full_path, curr_id))
                        L = len(full_path)
                        best_len = L if best_len is None else min(best_len, L)

                _, parents, _, _ = get_person_with_relatives(curr_id, headers)
                for p_id in parents:
                    if p_id in path: continue
                    new_path = path + [p_id]
                    if _add_path_variant(visited1, p_id, new_path):
                        q1.append((p_id, new_path))
                        expanded += 1
                        if expanded >= max_nodes: break
                if expanded >= max_nodes: break
            if expanded >= max_nodes: break

        # Processa a fronteira 2 apenas se a fila não estiver vazia
        if q2:
            q2_size = len(q2)
            for _ in range(q2_size):
                curr_id, path = q2.popleft()
                if DEBUG_FS: print(f"[DEBUG][Lado 2] Processando: {curr_id}")

                if curr_id in visited1:
                    if DEBUG_FS: print(f"[DEBUG][ENCONTRO!] ID {curr_id} achado em ambas as buscas.")
                    for p1 in visited1[curr_id]:
                        full_path = p1 + path[::-1][1:]
                        paths_with_ancestors.append((full_path, curr_id))
                        L = len(full_path)
                        best_len = L if best_len is None else min(best_len, L)

                _, parents, _, _ = get_person_with_relatives(curr_id, headers)
                for p_id in parents:
                    if p_id in path: continue
                    new_path = path + [p_id]
                    if _add_path_variant(visited2, p_id, new_path):
                        q2.append((p_id, new_path))
                        expanded += 1
                        if expanded >= max_nodes: break
                if expanded >= max_nodes: break
            if expanded >= max_nodes: break
        # >>> FIM DA CORREÇÃO <<<

        depth += 1
        if best_len is not None and depth > best_len + 2:
            break

    final_paths = post_process_paths(paths_with_ancestors, headers)
    if DEBUG_FS:
        print(f"[PathFinder] depth={depth} expanded={expanded} best_len={best_len} "
              f"paths_raw={len(paths_with_ancestors)} paths_final={len(final_paths)}")
    return final_paths

# -------------------------------------------------------------
# Mermaid (IDs saneados + labels escapados) — caixa única p/ casal AC
# -------------------------------------------------------------
def generate_mermaid_graph(path_details):
    """Gera árvore TD com o casal ancestral em comum em UMA caixa."""
    def sid(raw: str) -> str:
        raw = str(raw).replace('+', '_').replace('-', '_')
        s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in raw)
        if not s or not s[0].isalpha():
            s = 'N_' + s
        return s

    def lab(txt: str) -> str:
        txt = str(txt).replace('↔', ' & ').replace('\n', ' ').replace('\r', ' ')
        return txt.replace('"', r'\"')

    lines = ["flowchart TD"]
    seen = set()

    ac_idx = next((i for i, n in enumerate(path_details) if n.get('is_common_ancestor')), -1)
    if ac_idx == -1:
        # fallback: linha simples
        ids = []
        for n in path_details:
            nid = sid(n['id'])
            if nid not in seen:
                lines.append(f'{nid}["{lab(n["name"])}"]'); seen.add(nid)
            ids.append(nid)
        for i in range(len(ids) - 1):
            lines.append(f'{ids[i]} --> {ids[i+1]}')
        lines.append(f'style {ids[0]} fill:#e8f5e9,stroke:#66bb6a,stroke-width:2px')
        lines.append(f'style {ids[-1]} fill:#ffebee,stroke:#ef5350,stroke-width:2px')
        return "\n".join(lines)

    ac_node = path_details[ac_idx]

    # cria UM nó para o casal ancestral (ou nó único se não for casal)
    if ac_node.get('is_couple'):
        sp_ids = ac_node['id'].split('+')
        couple_raw_id = 'C_' + '_'.join(sorted(sp_ids))
        ac_mermaid_id = sid(couple_raw_id)
        lines.append(f'{ac_mermaid_id}["{lab(ac_node["name"])}"]')
        lines.append(f'style {ac_mermaid_id} fill:#fff9c4,stroke:#fbc02d,stroke-width:2px')
    else:
        ac_mermaid_id = sid(ac_node['id'])
        lines.append(f'{ac_mermaid_id}["{lab(ac_node["name"])}"]')
        lines.append(f'style {ac_mermaid_id} fill:#fff9c4,stroke:#fbc02d,stroke-width:2px')
    seen.add(ac_mermaid_id)

    def node_id(n) -> str:
        if n.get('is_common_ancestor') and ac_node.get('is_couple'):
            return ac_mermaid_id
        nid_raw = n['id'].split('+')[0] if n.get('is_couple') else n['id']
        nid = sid(nid_raw)
        if nid not in seen:
            lines.append(f'{nid}["{lab(n["name"])}"]'); seen.add(nid)
        return nid

    left_branch  = list(reversed(path_details[:ac_idx+1]))  # AC -> ... -> P1
    right_branch = path_details[ac_idx:]                    # AC -> ... -> P2

    def connect_branch(branch):
        prev = ac_mermaid_id
        for n in branch[1:]:
            cur = node_id(n)
            lines.append(f'{prev} --> {cur}')
            prev = cur

    connect_branch(left_branch)
    connect_branch(right_branch)

    # estilos folhas
    p1_leaf = node_id(path_details[0])
    p2_leaf = node_id(path_details[-1])
    lines.append(f'style {p1_leaf} fill:#e8f5e9,stroke:#66bb6a,stroke-width:2px')
    lines.append(f'style {p2_leaf} fill:#ffebee,stroke:#ef5350,stroke-width:2px')

    return "\n".join(lines)

# ---------- Grau de parentesco (PT-BR) ----------
_ORD_PT = {1: "1º", 2: "2º", 3: "3º", 4: "4º", 5: "5º", 6: "6º", 7: "7º", 8: "8º", 9: "9º", 10: "10º"}
def _ord_pt(n: int) -> str:
    return _ORD_PT.get(n, f"{n}º")

def relationship_label(d1: int, d2: int) -> str:
    if d1 == 0 and d2 > 0:
        return f"Ascendência direta ({d2} geração{'s' if d2 > 1 else ''})"
    if d2 == 0 and d1 > 0:
        return f"Descendência direta ({d1} geração{'s' if d1 > 1 else ''})"
    if d1 == 1 and d2 == 1:
        return "Irmãos(ãs)"
    c = min(d1, d2) - 1
    r = abs(d1 - d2)
    if c >= 1:
        base = f"{_ord_pt(c)} primo"
        return f"{base}, {r}x removido" if r > 0 else base
    if r == 1:
        return "Tio/Tia ↔ Sobrinho(a)"
    return f"Parentes colaterais ({r}x removido)"

def ensure_degree_labels_in_paths(paths: list):
    """Garante que cada caminho tenha p['degree_label'] mesmo em snapshots antigos."""
    for p in paths or []:
        if p.get("degree_label"):
            continue
        nodes = p.get("nodes") or []
        if not nodes:
            continue
        ac_idx = next((i for i, n in enumerate(nodes) if n.get("is_common_ancestor")), None)
        if ac_idx is None:
            continue
        d1 = ac_idx
        d2 = (len(nodes) - 1) - ac_idx
        p["degree_label"] = relationship_label(d1, d2)


def _share_path(slug: str) -> pathlib.Path:
    return SHARE_DIR / f"{slug}.json"

# -------------------------------------------------------------
# Relationship Finder (somente-leitura, token UNAUTH)
# -------------------------------------------------------------
def rf_path_unauth(p1: str, p2: str):
    """
    Tenta o Relationship Finder com token 'unauthenticated_session'.
    Retorna dict com diagnóstico:
      {"ok": True, "ids": [...], "common": "..."}
      {"ok": False, "reason": "no_token|http|exception", "status": ..., "detail": "..."}
    """
    token = get_unauth_token()
    if not token:
        return {"ok": False, "reason": "no_token"}

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "PathFinder/2.0",
    }
    url = f"{API_BASE_URL}/platform/tree/persons/{p1}/relationships/{p2}?personDetails=true"
    try:
        r = session_http.get(url, headers=headers, verify=False, timeout=DEFAULT_TIMEOUT)
        if r.status_code != 200:
            return {"ok": False, "reason": "http", "status": r.status_code, "detail": r.text[:200]}
        j = r.json()
        persons = j.get("persons") or []
        ids = [p.get("id") for p in persons if p.get("id")]
        if not ids or ids[0] != p1 or ids[-1] != p2:
            return {"ok": False, "reason": "no_path"}

        common = None
        for p in persons:
            disp = p.get("displayProperties") or {}
            if disp.get("role") == "commonAncestor":
                common = p.get("id")
                break
        if not common and len(ids) >= 3:
            common = ids[len(ids)//2]

        return {"ok": True, "ids": ids, "common": common}
    except Exception as e:
        return {"ok": False, "reason": "exception", "detail": str(e)[:200]}


def build_path_details_from_ids(ids, common_id, headers):
    details = []
    for pid in ids:
        name = get_person_name(pid, headers)
        details.append({
            "id": pid,
            "name": name,
            "is_couple": False,
            "is_common_ancestor": (pid == common_id),
        })
    return details

# -------------------------------------------------------------
# Flask routes
# -------------------------------------------------------------
@app.route("/")
def index():
    if "access_token" not in session:
        return render_template("index.html")
    return redirect(url_for("search"))

@app.route("/login")
def login():
    return redirect(build_auth_url())

@app.route("/callback")
def callback():
    code = request.args.get("code")
    if not code:
        return "Código não informado.", 400
    token_data = exchange_code_for_token(code)
    if not token_data:
        return "Falha ao obter token.", 400
    session["access_token"] = token_data.get("access_token")
    return redirect(url_for("search"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/search", methods=["GET", "POST"])
def search():
    if "access_token" not in session:
        return redirect(url_for("index"))

    headers = get_headers()
    if request.method == "POST":
        person1_id = (request.form.get("person1_id") or "").strip().upper()
        person2_id = (request.form.get("person2_id") or "").strip().upper()
        max_depth  = int(request.form.get("max_depth") or 8)

        if not person1_id or not person2_id:
            return render_template("search.html", error="Informe os dois IDs.", max_depth=max_depth)

        t0 = time.time()
        paths = find_paths(person1_id, person2_id, headers, max_depth=max_depth)
        elapsed = time.time() - t0

        paths_vm = []
        for raw_path, ancestor in paths:
            path_details = []
            for raw in raw_path:
                if isinstance(raw, tuple):
                    a, b = raw
                    an = get_person_name(a, headers) or a
                    bn = get_person_name(b, headers) or b
                    node = {
                        "id": f"{a}+{b}",
                        "name": f"{an} & {bn}",
                        "is_couple": True,
                        "is_common_ancestor": (isinstance(ancestor, tuple) and set(raw) == set(ancestor)),
                    }
                else:
                    nm = get_person_name(raw, headers) or raw
                    node = {
                        "id": raw,
                        "name": nm,
                        "is_couple": False,
                        "is_common_ancestor": (not isinstance(ancestor, tuple) and raw == ancestor),
                    }
                path_details.append(node)

            p1_name = path_details[0]["name"] if path_details else person1_id
            p2_name = path_details[-1]["name"] if path_details else person2_id
            mermaid_data = generate_mermaid_graph(path_details)

            ac_idx = next((i for i, n in enumerate(path_details) if n.get("is_common_ancestor")), None)
            deg_label = None
            if ac_idx is not None:
                d1 = ac_idx
                d2 = (len(path_details) - 1) - ac_idx
                deg_label = relationship_label(d1, d2)

            paths_vm.append({
                "nodes": path_details,
                "p1_name": p1_name,
                "p2_name": p2_name,
                "mermaid_data": mermaid_data,
                "degree_label": deg_label,
            })

        if not paths_vm:
            return render_template(
                "search.html",
                error=f"Nenhum caminho encontrado (tempo {elapsed:.2f}s). "
                      f"Tente aumentar a 'Profundidade Máxima' ou verifique os IDs.",
                person1_id=person1_id,
                person2_id=person2_id,
                max_depth=max_depth
            )

        return render_template(
            "search.html",
            paths=paths_vm,
            person1_id=person1_id,
            person2_id=person2_id,
            max_depth=max_depth,
            elapsed=elapsed
        )

    # GET
    return render_template("search.html", max_depth=8)

# ----- Link público leve (sem login) -----
@app.get("/view")
def view_public():
    """
    Ex: /view?p1=K123-ABC&p2=L987-XYZ&d=8
    1) tenta RF público
    2) se falhar, tenta BFS público (quando perfis são públicos)
    """
    p1 = (request.args.get("p1") or "").strip().upper()
    p2 = (request.args.get("p2") or "").strip().upper()
    d  = int(request.args.get("d") or 8)
    if not p1 or not p2:
        return "Parâmetros p1 e p2 são obrigatórios.", 400

    # 1) RF público
    rf = rf_path_unauth(p1, p2)
    if rf.get("ok"):
        token = get_unauth_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "PathFinder/2.0",
        }
        path_details = build_path_details_from_ids(rf["ids"], rf["common"], headers)
        mermaid_data = generate_mermaid_graph(path_details)
        data = {
            "person1_id": p1,
            "person2_id": p2,
            "max_depth": d,
            "paths": [{
                "p1_name": path_details[0]["name"],
                "p2_name": path_details[-1]["name"],
                "mermaid_data": mermaid_data
            }]
        }
        return render_template("share.html", data=data, slug=None, error=None)

    # 2) Fallback: BFS público (pode funcionar para perfis falecidos)
    token = get_unauth_token()
    if token:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "PathFinder/2.0",
        }
        try:
            paths = find_paths(p1, p2, headers, max_depth=d)
            if paths:
                raw_path, ancestor = paths[0]
                # monta details como no /search
                path_details = []
                for raw in raw_path:
                    if isinstance(raw, tuple):
                        a, b = raw
                        an = get_person_name(a, headers) or a
                        bn = get_person_name(b, headers) or b
                        node = {
                            "id": f"{a}+{b}",
                            "name": f"{an} & {bn}",
                            "is_couple": True,
                            "is_common_ancestor": (isinstance(ancestor, tuple) and set(raw) == set(ancestor)),
                        }
                    else:
                        nm = get_person_name(raw, headers) or raw
                        node = {
                            "id": raw,
                            "name": nm,
                            "is_couple": False,
                            "is_common_ancestor": (not isinstance(ancestor, tuple) and raw == ancestor),
                        }
                    path_details.append(node)

                mermaid_data = generate_mermaid_graph(path_details)
                data = {
                    "person1_id": p1,
                    "person2_id": p2,
                    "max_depth": d,
                    "paths": [{
                        "p1_name": path_details[0]["name"],
                        "p2_name": path_details[-1]["name"],
                        "mermaid_data": mermaid_data
                    }]
                }
                return render_template("share.html", data=data, slug=None, error=None)
        except Exception as e:
            if DEBUG_FS: print("[/view BFS fallback] EXC:", e)

    # Mensagem amigável conforme o motivo
    reason = rf.get("reason")
    status = rf.get("status")
    msg = "Não foi possível calcular o relacionamento com acesso público."
    if status in (401, 403) or reason in ("no_token", "http"):
        msg += " Pelo menos um dos perfis parece privado (pessoa viva) ou seu app não tem permissão pública."
    elif reason == "no_path":
        msg += " O Relationship Finder público não retornou caminho entre esses IDs."
    # Render mínimo
    data = {"person1_id": p1, "person2_id": p2, "max_depth": d, "paths": []}
    return render_template("share.html", data=data, slug=None, error=msg)

# ----- Snapshot de compartilhamento (sem login para visualizar) -----
@app.post("/api/share")
def create_share():
    """
    Recebe { person1_id, person2_id, max_depth, paths } e grava um snapshot
    em JSON. Devolve a URL pública /share/<slug>.
    """
    try:
        payload = request.get_json(force=True, silent=False) or {}
    except Exception:
        return jsonify({"ok": False, "error": "JSON inválido"}), 400

    required = ("person1_id", "person2_id", "max_depth", "paths")
    if not all(k in payload for k in required):
        return jsonify({"ok": False, "error": "Campos obrigatórios ausentes"}), 400

    payload["_meta"] = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "version": "pf-share-1",
    }

    slug = secrets.token_urlsafe(6)
    fpath = _share_path(slug)
    try:
        fpath.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        return jsonify({"ok": False, "error": f"Falha ao salvar: {e}"}), 500

    share_url = url_for("view_share", slug=slug, _external=True)
    return jsonify({"ok": True, "url": share_url, "slug": slug})

@app.get("/share/<slug>")
def view_share(slug):
    """
    Página somente-leitura que exibe o snapshot (sem login/token).
    """
    fpath = _share_path(slug)
    if not fpath.exists():
        abort(404)
    try:
        data = json.loads(fpath.read_text(encoding="utf-8"))
    except Exception:
        abort(500)
    
     # >>> NOVO: preenche degree_label se faltar no snapshot
    ensure_degree_labels_in_paths(data.get("paths"))
    
    return render_template("share.html", data=data, slug=slug, error=None)

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
if __name__ == "__main__":
    # Em produção, rode atrás de um proxy HTTPS (gunicorn/uvicorn + reverse-proxy)
    debug = DEBUG_FS
    ssl_ctx = None
    # Para dev local HTTPS (se tiver cert.pem/key.pem)
    cert_file = os.getenv("SSL_CERT_FILE", "cert.pem")
    key_file  = os.getenv("SSL_KEY_FILE", "key.pem")
    if os.path.exists(cert_file) and os.path.exists(key_file):
        ssl_ctx = (cert_file, key_file)
    app.run(debug=debug, ssl_context=ssl_ctx)

