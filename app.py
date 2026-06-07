import math
import random
import time
import threading
import json
import os
import requests
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="VectorDB Engine (Python)")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_index():
    return FileResponse("index.html")

DIMS = 16

# =====================================================================
#  DATA TYPES
# =====================================================================

@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: List[float]

# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a: List[float], b: List[float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def cosine(a: List[float], b: List[float]) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    dot = np.dot(a, b)
    return float(1.0 - dot / (na * nb))

def manhattan(a: List[float], b: List[float]) -> float:
    return float(np.sum(np.abs(np.array(a) - np.array(b))))

def get_dist_fn(m: str):
    if m == "cosine":
        return cosine
    if m == "manhattan":
        return manhattan
    return euclidean

# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    def __init__(self):
        self.items: List[VectorItem] = []

    def insert(self, v: VectorItem):
        self.items.append(v)

    def knn(self, q: List[float], k: int, dist_fn) -> List[Tuple[float, int]]:
        results = []
        for v in self.items:
            results.append((dist_fn(q, v.emb), v.id))
        results.sort(key=lambda x: x[0])
        return results[:k]

    def remove(self, id: int):
        self.items = [v for v in self.items if v.id != id]

# =====================================================================
#  KD-TREE
# =====================================================================

class KDNode:
    def __init__(self, item: VectorItem):
        self.item = item
        self.left = None
        self.right = None

class KDTree:
    def __init__(self, dims: int):
        self.root = None
        self.dims = dims

    def insert(self, v: VectorItem):
        def _ins(n, v, d):
            if not n:
                return KDNode(v)
            ax = d % self.dims
            if v.emb[ax] < n.item.emb[ax]:
                n.left = _ins(n.left, v, d + 1)
            else:
                n.right = _ins(n.right, v, d + 1)
            return n
        self.root = _ins(self.root, v, 0)

    def knn(self, q: List[float], k: int, dist_fn) -> List[Tuple[float, int]]:
        import heapq
        heap = [] # max-heap of (distance, id)

        def _knn(n, q, k, d):
            if not n:
                return
            dn = dist_fn(q, n.item.emb)
            if len(heap) < k or dn < -heap[0][0]:
                heapq.heappush(heap, (-dn, n.item.id))
                if len(heap) > k:
                    heapq.heappop(heap)
            
            ax = d % self.dims
            diff = q[ax] - n.item.emb[ax]
            closer = n.left if diff < 0 else n.right
            farther = n.right if diff < 0 else n.left
            
            _knn(closer, q, k, d + 1)
            if len(heap) < k or abs(diff) < -heap[0][0]:
                _knn(farther, q, k, d + 1)

        _knn(self.root, q, k, 0)
        res = [(-d, id) for d, id in heap]
        res.sort()
        return res

    def rebuild(self, items: List[VectorItem]):
        self.root = None
        for v in items:
            self.insert(v)

# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

class HNSW:
    class Node:
        def __init__(self, item: VectorItem, lvl: int):
            self.item = item
            self.max_lyr = lvl
            self.nbrs = [[] for _ in range(lvl + 1)]

    def __init__(self, m: int = 16, ef_build: int = 200):
        self.G: Dict[int, HNSW.Node] = {}
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(float(m))
        self.top_layer = -1
        self.entry_pt = -1
        self.rng = random.Random(42)

    def _rand_level(self) -> int:
        u = self.rng.random()
        if u == 0: u = 1e-9
        return int(math.floor(-math.log(u) * self.mL))

    def _search_layer(self, q: List[float], ep: int, ef: int, lyr: int, dist_fn) -> List[Tuple[float, int]]:
        import heapq
        vis = {ep}
        cands = [(dist_fn(q, self.G[ep].item.emb), ep)] # min-heap
        found = [(-cands[0][0], ep)] # max-heap

        while cands:
            cd, cid = heapq.heappop(cands)
            if len(found) >= ef and cd > -found[0][0]:
                break
            
            if lyr >= len(self.G[cid].nbrs):
                continue
                
            for nid in self.G[cid].nbrs[lyr]:
                if nid in vis or nid not in self.G:
                    continue
                vis.add(nid)
                nd = dist_fn(q, self.G[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)
        
        res = [(-d, id) for d, id in found]
        res.sort()
        return res

    def _select_nbrs(self, cands: List[Tuple[float, int]], max_m: int) -> List[int]:
        return [c[1] for c in cands[:max_m]]

    def insert(self, item: VectorItem, dist_fn):
        id = item.id
        lvl = self._rand_level()
        self.G[id] = HNSW.Node(item, lvl)

        if self.entry_pt == -1:
            self.entry_pt = id
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            W = self._search_layer(item.emb, ep, 1, lc, dist_fn)
            if W:
                ep = W[0][1]
        
        for lc in range(min(self.top_layer, lvl), -1, -1):
            W = self._search_layer(item.emb, ep, self.ef_build, lc, dist_fn)
            max_m = self.M0 if lc == 0 else self.M
            sel = self._select_nbrs(W, max_m)
            self.G[id].nbrs[lc] = sel

            for nid in sel:
                if nid not in self.G: continue
                if len(self.G[nid].nbrs) <= lc:
                    self.G[nid].nbrs.extend([[] for _ in range(lc - len(self.G[nid].nbrs) + 1)])
                conn = self.G[nid].nbrs[lc]
                conn.append(id)
                if len(conn) > max_m:
                    ds = []
                    for c in conn:
                        if c in self.G:
                            ds.append((dist_fn(self.G[nid].item.emb, self.G[c].item.emb), c))
                    ds.sort()
                    self.G[nid].nbrs[lc] = [d[1] for d in ds[:max_m]]
            
            if W:
                ep = W[0][1]
        
        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = id

    def knn(self, q: List[float], k: int, ef: int, dist_fn) -> List[Tuple[float, int]]:
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            W = self._search_layer(q, ep, 1, lc, dist_fn)
            if W:
                ep = W[0][1]
        W = self._search_layer(q, ep, max(ef, k), 0, dist_fn)
        return W[:k]

    def remove(self, id: int):
        if id not in self.G: return
        for nid in self.G:
            for layer in self.G[nid].nbrs:
                if id in layer:
                    layer.remove(id)
        if self.entry_pt == id:
            self.entry_pt = -1
            for nid in self.G:
                if nid != id:
                    self.entry_pt = nid
                    break
        del self.G[id]

    def get_info(self):
        max_l = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes = []
        edges = []
        for id, nd in self.G.items():
            nodes.append({
                "id": id,
                "metadata": nd.item.metadata,
                "category": nd.item.category,
                "maxLyr": nd.max_lyr
            })
            for lc in range(min(nd.max_lyr + 1, max_l)):
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if id < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": id, "dst": nid, "lyr": lc})
        return {
            "topLayer": self.top_layer,
            "nodeCount": len(self.G),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges
        }

# =====================================================================
#  VECTOR DATABASE
# =====================================================================

class VectorDB:
    def __init__(self, dims: int):
        self.dims = dims
        self.store: Dict[int, VectorItem] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(16, 200)
        self.mu = threading.Lock()
        self.next_id = 1

    def insert(self, meta: str, cat: str, emb: List[float], dist_fn) -> int:
        with self.mu:
            v = VectorItem(self.next_id, meta, cat, emb)
            self.next_id += 1
            self.store[v.id] = v
            self.bf.insert(v)
            self.kdt.insert(v)
            self.hnsw.insert(v, dist_fn)
            return v.id

    def remove(self, id: int) -> bool:
        with self.mu:
            if id not in self.store: return False
            del self.store[id]
            self.bf.remove(id)
            self.hnsw.remove(id)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, q: List[float], k: int, metric: str, algo: str):
        with self.mu:
            dfn = get_dist_fn(metric)
            t0 = time.perf_counter_ns()
            
            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dfn)
            else:
                raw = self.hnsw.knn(q, k, 50, dfn)
            
            us = (time.perf_counter_ns() - t0) // 1000
            
            hits = []
            for d, id in raw:
                if id in self.store:
                    v = self.store[id]
                    hits.append({
                        "id": v.id,
                        "metadata": v.metadata,
                        "category": v.category,
                        "embedding": v.emb,
                        "distance": d
                    })
            return {"results": hits, "latencyUs": us, "algo": algo, "metric": metric}

    def benchmark(self, q: List[float], k: int, metric: str):
        with self.mu:
            dfn = get_dist_fn(metric)
            def time_it(fn):
                t = time.perf_counter_ns()
                fn()
                return (time.perf_counter_ns() - t) // 1000
            
            return {
                "bruteforceUs": time_it(lambda: self.bf.knn(q, k, dfn)),
                "kdtreeUs": time_it(lambda: self.kdt.knn(q, k, dfn)),
                "hnswUs": time_it(lambda: self.hnsw.knn(q, k, 50, dfn)),
                "n": len(self.store)
            }

# =====================================================================
#  OLLAMA CLIENT
# =====================================================================

class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.base_url = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2"

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if r.status_code == 200:
                return True
        except:
            pass
        if os.environ.get("HF_TOKEN"):
            return True
        return False

    def embed(self, text: str) -> List[float]:
        try:
            r = requests.post(f"{self.base_url}/api/embeddings", 
                             json={"model": self.embed_model, "prompt": text},
                             timeout=5)
            if r.status_code == 200:
                return r.json().get("embedding", [])
        except:
            pass
        
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            try:
                headers = {"Authorization": f"Bearer {hf_token}"}
                r = requests.post(
                    "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
                    headers=headers,
                    json={"inputs": text},
                    timeout=15
                )
                if r.status_code == 200:
                    res = r.json()
                    def _parse_embedding(val) -> List[float]:
                        if not isinstance(val, list) or len(val) == 0:
                            return []
                        if isinstance(val[0], (int, float)):
                            return [float(x) for x in val]
                        if isinstance(val[0], list):
                            return _parse_embedding(val[0])
                        return []
                    parsed = _parse_embedding(res)
                    if parsed:
                        return parsed
            except Exception as e:
                print(f"HF embed error: {e}")
        return []

    def generate(self, prompt: str) -> str:
        try:
            r = requests.post(f"{self.base_url}/api/generate",
                             json={"model": self.gen_model, "prompt": prompt, "stream": False},
                             timeout=30)
            if r.status_code == 200:
                return r.json().get("response", "")
        except:
            pass

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            models = ["Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"]
            for model_name in models:
                try:
                    headers = {"Authorization": f"Bearer {hf_token}"}
                    payload = {
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": 500,
                            "return_full_text": False
                        }
                    }
                    r = requests.post(
                        f"https://api-inference.huggingface.co/models/{model_name}",
                        headers=headers,
                        json=payload,
                        timeout=30
                    )
                    if r.status_code == 200:
                        res = r.json()
                        if isinstance(res, list) and len(res) > 0:
                            return res[0].get("generated_text", "")
                except Exception as e:
                    print(f"HF generate error for {model_name}: {e}")
            
            return "ERROR: Hugging Face Inference API models timed out or were unavailable."
        
        return "ERROR: Ollama unavailable. Run: ollama serve"

# =====================================================================
#  DOCUMENT DATABASE
# =====================================================================

@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: List[float]

class DocumentDB:
    def __init__(self):
        self.store: Dict[int, DocItem] = {}
        self.hnsw = HNSW(16, 200)
        self.bf = BruteForce()
        self.mu = threading.Lock()
        self.next_id = 1
        self.dims = 0

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        with self.mu:
            if self.dims == 0: self.dims = len(emb)
            item = DocItem(self.next_id, title, text, emb)
            self.next_id += 1
            self.store[item.id] = item
            vi = VectorItem(item.id, title, "doc", emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item.id

    def search(self, q: List[float], k: int, max_dist: float = 0.7) -> List[Tuple[float, DocItem]]:
        with self.mu:
            if not self.store: return []
            raw = self.bf.knn(q, k, cosine) if len(self.store) < 10 else self.hnsw.knn(q, k, 50, cosine)
            out = []
            for d, id in raw:
                if id in self.store and d <= max_dist:
                    out.append((d, self.store[id]))
            return out

    def remove(self, id: int) -> bool:
        with self.mu:
            if id not in self.store: return False
            del self.store[id]
            self.hnsw.remove(id)
            self.bf.remove(id)
            return True

# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> List[str]:
    words = text.split()
    if not words: return []
    if len(words) <= chunk_words: return [text]
    
    chunks = []
    step = chunk_words - overlap_words
    for i in range(0, len(words), step):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words): break
    return chunks

# =====================================================================
#  GLOBAL STATE
# =====================================================================

db = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()

def load_demo():
    dist = get_dist_fn("cosine")
    demo_data = [
        ("Linked List: nodes connected by pointers", "cs", [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]),
        ("Binary Search Tree: O(log n) search and insert", "cs", [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10]),
        ("Dynamic Programming: memoization overlapping subproblems", "cs", [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07]),
        ("Graph BFS and DFS: breadth and depth first traversal", "cs", [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07]),
        ("Hash Table: O(1) lookup with collision chaining", "cs", [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09]),
        ("Calculus: derivatives integrals and limits", "math", [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10]),
        ("Linear Algebra: matrices eigenvalues eigenvectors", "math", [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09]),
        ("Probability: distributions random variables Bayes theorem", "math", [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08]),
        ("Number Theory: primes modular arithmetic RSA cryptography", "math", [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06]),
        ("Combinatorics: permutations combinations generating functions", "math", [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07]),
        ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food", [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07]),
        ("Sushi: vinegared rice raw fish and nori rolls", "food", [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]),
        ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food", [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06]),
        ("Tacos: corn tortillas with carnitas salsa and cilantro", "food", [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09]),
        ("Croissant: laminated pastry with buttery flaky layers", "food", [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06]),
        ("Basketball: fast-paced shooting dribbling slam dunks", "sports", [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]),
        ("Football: tackles touchdowns field goals and strategy", "sports", [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76]),
        ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports", [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82]),
        ("Chess: openings endgames tactics strategic board game", "sports", [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90]),
        ("Swimming: butterfly freestyle backstroke Olympic competition", "sports", [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80]),
    ]
    for meta, cat, emb in demo_data:
        db.insert(meta, cat, emb, dist)

load_demo()

# =====================================================================
#  API ENDPOINTS
# =====================================================================

class InsertRequest(BaseModel):
    metadata: str
    category: str
    embedding: List[float]

class DocInsertRequest(BaseModel):
    title: str
    text: str

class AskRequest(BaseModel):
    question: str
    k: int = 3

@app.get("/status")
async def get_status():
    try:
        r = requests.get(f"{ollama.base_url}/api/tags", timeout=2)
        local_ok = r.status_code == 200
    except:
        local_ok = False
        
    hf_ok = bool(os.environ.get("HF_TOKEN"))
    return {
        "ollamaAvailable": local_ok or hf_ok,
        "embedModel": ollama.embed_model if local_ok else "HF-MiniLM",
        "genModel": ollama.gen_model if local_ok else "HF-Qwen2.5",
        "docCount": len(doc_db.store),
        "docDims": doc_db.dims,
        "vectorCount": len(db.store),
        "isFallback": not local_ok and hf_ok
    }

@app.get("/items")
async def get_items():
    return [{"id": v.id, "metadata": v.metadata, "category": v.category, "embedding": v.emb} for v in db.store.values()]

@app.get("/search")
async def search(v: str, k: int = 5, metric: str = "cosine", algo: str = "hnsw"):
    try:
        q = [float(x) for x in v.split(",")]
    except:
        raise HTTPException(status_code=400, detail="Invalid vector format")
    
    if len(q) != DIMS:
        raise HTTPException(status_code=400, detail=f"Need {DIMS}D vector")
    
    return db.search(q, k, metric, algo)

@app.post("/insert")
async def insert(req: InsertRequest):
    dist = get_dist_fn("cosine")
    id = db.insert(req.metadata, req.category, req.embedding, dist)
    return {"id": id}

@app.delete("/delete/{id}")
async def delete_item(id: int):
    if db.remove(id):
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Item not found")

@app.get("/benchmark")
async def benchmark(v: str, k: int = 5, metric: str = "cosine"):
    try:
        q = [float(x) for x in v.split(",")]
    except:
        raise HTTPException(status_code=400, detail="Invalid vector format")
    return db.benchmark(q, k, metric)

@app.get("/hnsw-info")
async def hnsw_info():
    return db.hnsw.get_info()

@app.post("/doc/insert")
async def doc_insert(req: DocInsertRequest):
    if not ollama.is_available():
        return {"error": "Ollama unavailable"}
    
    chunks = chunk_text(req.text)
    count = 0
    dims = 0
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if emb:
            doc_db.insert(f"{req.title} (Part {i+1})", chunk, emb)
            count += 1
            dims = len(emb)
    
    return {"chunks": count, "dims": dims}

@app.get("/doc/list")
async def doc_list():
    res = []
    for d in doc_db.store.values():
        res.append({
            "id": d.id,
            "title": d.title,
            "words": len(d.text.split()),
            "preview": d.text[:100] + "..."
        })
    return res

@app.delete("/doc/delete/{id}")
async def doc_delete(id: int):
    if doc_db.remove(id):
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Document not found")

@app.post("/doc/search")
async def doc_search(req: AskRequest):
    emb = ollama.embed(req.question)
    if not emb:
        return {"contexts": []}
    
    hits = doc_db.search(emb, req.k)
    contexts = []
    for d, item in hits:
        contexts.append({"title": item.title, "text": item.text, "distance": d})
    return {"contexts": contexts}

@app.post("/doc/ask")
async def doc_ask(req: AskRequest):
    if not ollama.is_available():
        return {"error": "Ollama unavailable"}
    
    emb = ollama.embed(req.question)
    if not emb:
        return {"error": "Could not embed question"}
    
    hits = doc_db.search(emb, req.k)
    
    context_text = ""
    contexts = []
    for i, (d, item) in enumerate(hits):
        context_text += f"[{i+1}] {item.title}:\n{item.text}\n\n"
        contexts.append({"id": item.id, "title": item.title, "text": item.text, "distance": d})
    
    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
        "Just answer the question naturally.\n\n"
        "Context:\n" + context_text +
        "Question: " + req.question + "\n\n"
        "Answer:"
    )
    
    response = ollama.generate(prompt)
    return {
        "answer": response,
        "model": ollama.gen_model,
        "contexts": contexts,
        "docCount": len(doc_db.store)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
