import torch
print("Torch version:", torch.__version__)
print("CUDA available?", torch.cuda.is_available(), "devices:", torch.cuda.device_count())

import json
import logging
import time
import pandas as pd
import pyodbc
import chromadb
import transformers

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    GenerationConfig
)
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

# suppress noisy HF logs
transformers.logging.set_verbosity_error()

# === Configurable parameters ===
MAX_ROUNDS = 3
FOLLOWUP_COUNT = 3
DEFAULT_OUTPUT = 'training.jsonl'
TRANSCRIPT_FILE = 'transcript.txt'
SCHEMA_SNIPPET_LEN = 500  # chars
MAX_TOKENS = 64

# Department to schema keyword mapping
DEPT_SCHEMA_FILTERS: Dict[str, List[str]] = {
    'Sales': ['Sales', 'InternetSales', 'ResellerSales'],
    'HR': ['Employee', 'Department'],
    'Finance': ['Finance', 'CurrencyRate'],
    'IT': ['CallCenter', 'SurveyResponse', 'Scenario'],
    'Marketing': ['Promotion', 'Product', 'SalesReason']
}

# Departments and their training parameters
departments = [
    {'name': 'Sales',   'rounds': MAX_ROUNDS, 'followups': FOLLOWUP_COUNT},
    {'name': 'HR',      'rounds': MAX_ROUNDS, 'followups': FOLLOWUP_COUNT},
    {'name': 'Finance', 'rounds': MAX_ROUNDS, 'followups': FOLLOWUP_COUNT},
    {'name': 'IT',      'rounds': MAX_ROUNDS, 'followups': FOLLOWUP_COUNT},
    {'name': 'Marketing','rounds': MAX_ROUNDS, 'followups': FOLLOWUP_COUNT},
]

# Department-specific initial questions
INITIAL_QUESTIONS: Dict[str, List[str]] = {
    'Sales': [
        "List all sales-related tables and their columns?",
        "Which tables and relationships affect monthly revenue reporting?",
        "What data types are used for sales metrics?"
    ],
    'HR': [
        "List all HR-related tables and their columns?",
        "Which tables capture employee-manager relationships?",
        "What data types are used in HR analytics?"
    ],
    'Finance': [
        "List all finance-related tables and their columns?",
        "Which tables and keys impact budgeting reports?",
        "What data types support financial aggregates?"
    ],
    'IT': [
        "List all IT tables related to system logs and metrics?",
        "Which relationships model server-to-application mappings?",
        "What data types store performance metrics?"
    ],
    'Marketing': [
        "List all marketing-related tables and their columns?",
        "Which tables track campaign and channel performance?",
        "What data types support marketing KPIs?"
    ]
}
DEFAULT_QUESTIONS = [
    "List all tables and their columns?",
    "Identify key relationships between tables?",
    "What data types are used?"
]

@dataclass
class QueryResult:
    question: str
    understanding: str
    sql_query: str
    results: pd.DataFrame
    explanation: str
    success: bool

class RagDB:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        logging.info("Initializing RagDB...")
        self.conn_str = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=localhost;"
            "DATABASE=AdventureWorksDW2022;"
            "Trusted_Connection=yes;"
        )
        self.device = 0 if torch.cuda.is_available() else -1
        logging.info(f"Using {'GPU' if self.device>=0 else 'CPU'} for inference")
        embed_device = 'cuda' if self.device>=0 else 'cpu'
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=embed_device)
        self._setup_llm()
        Path("vectordb").mkdir(exist_ok=True)
        client = chromadb.PersistentClient(path="vectordb")
        self.collection = client.get_or_create_collection("db_knowledge")
        self.db_schema: Dict[str, Any] = {}
        self._extract_db_schema()
        self._embed_schema()

    def _setup_llm(self):
        model_names = ["meta-llama/Llama-3-8b", "TheBloke/vicuna-7B-1.1-HF"] if self.device>=0 else ["gpt2"]
        for name in model_names:
            try:
                logging.info(f"Loading model {name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False)
                kwargs = {}
                if self.device>=0:
                    kwargs = {"device_map":"auto", "torch_dtype":torch.float16}
                    if 'Llama-3' in name:
                        kwargs['load_in_4bit'] = True
                self.llm_model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
                logging.info(f"Loaded {name}")
                break
            except Exception as e:
                logging.warning(f"Failed to load {name}: {e}")
        gen_conf = GenerationConfig(max_new_tokens=MAX_TOKENS, temperature=0.7, do_sample=True)
        self.text_generator = pipeline(
            "text-generation", model=self.llm_model, tokenizer=self.tokenizer,
            return_full_text=False, generation_config=gen_conf
        )

    def _extract_db_schema(self):
        conn = pyodbc.connect(self.conn_str)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE='BASE TABLE' AND TABLE_SCHEMA='dbo'
              AND (TABLE_NAME LIKE 'Dim%' OR TABLE_NAME LIKE 'Fact%')
            ORDER BY TABLE_NAME
            """
        )
        for (table,) in cur.fetchall():
            cur.execute(f"SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='{table}'")
            cols = cur.fetchall()
            cur.execute(f"SELECT TOP 1 * FROM dbo.{table}")
            sample = cur.fetchall(); names=[d[0] for d in cur.description]
            cur.execute(f"SELECT COUNT(*) FROM dbo.{table}")
            cnt = cur.fetchone()[0]
            self.db_schema[table] = {
                'columns': [(c[0],c[1],c[2]) for c in cols],
                'sample': [dict(zip(names,row)) for row in sample],
                'row_count': cnt
            }
            logging.info(f"Schema: {table} cols {len(cols)}, rows {cnt}")
        conn.close()

    def _embed_schema(self):
        if self.collection.count() > 0:
            return
        docs, metas, ids = [], [], []
        for tbl, info in self.db_schema.items():
            docs.append(f"{tbl}: columns {[c[0] for c in info['columns']]} rows {info['row_count']}")
            metas.append({'table': tbl}); ids.append(f"tbl_{tbl}")
            if info['sample']:
                docs.append(f"Sample {tbl}: {info['sample'][0]}")
                metas.append({'table':tbl, 'type':'sample'}); ids.append(f"samp_{tbl}")
        embs = self.embedder.encode(docs, batch_size=64, show_progress_bar=True)
        self.collection.add(embeddings=embs.tolist(), documents=docs, metadatas=metas, ids=ids)
        logging.info("Embedded database schema into vector store")

    def _generate(self, prompt: str) -> str:
        start = time.time()
        out = self.text_generator(prompt)[0]['generated_text']
        logging.debug(f"Generation took {time.time()-start:.2f}s")
        return out

    def build_initial_questions(self, dept: str) -> List[str]:
        return INITIAL_QUESTIONS.get(dept, DEFAULT_QUESTIONS)

    def _get_schema_snippet(self, dept: str) -> str:
        keywords = DEPT_SCHEMA_FILTERS.get(dept, [])
        filtered = {t: self.db_schema[t] for t in self.db_schema if any(kw in t for kw in keywords)}
        if not filtered:
            filtered = self.db_schema
        return json.dumps(filtered, default=str)[:SCHEMA_SNIPPET_LEN]

    def generate_followups(self, question: str, answer: str, memory: str,
                           dept: str, count: int) -> List[str]:
        prompt = (
            f"You are a {dept} domain expert. Based on Memory:\n{memory}\nQ: {question}\nA: {answer}\n"
            f"Propose {count} concise, on-topic follow-up questions ending with '?'."
        )
        raw = self._generate(prompt)
        try:
            arr = json.loads(raw[raw.find('['): raw.rfind(']')+1])
        except:
            arr = [ln.strip('- ').strip() for ln in raw.splitlines() if ln.strip()]
        qs = []
        for item in arr:
            q = item if isinstance(item, str) else json.dumps(item)
            q = q.strip().strip('"')
            if q.endswith('?') and 10 <= len(q) <= 100:
                qs.append(q)
            if len(qs) == count:
                break
        return qs

    def generate_training_data(self):
        for f in (DEFAULT_OUTPUT, TRANSCRIPT_FILE):
            try: Path(f).unlink()
            except: pass
        transcript = []
        for cfg in departments:
            dept, rounds, fups = cfg['name'], cfg['rounds'], cfg['followups']
            qs = self.build_initial_questions(dept)
            memory = ""
            for r in range(1, rounds+1):
                print(f"\n=== Dept: {dept}, Round: {r}/{rounds} ===")
                transcript.append(f"=== Dept: {dept}, Round: {r}/{rounds} ===")
                snippet = self._get_schema_snippet(dept)
                next_qs = []
                for q in qs:
                    print(f"Q: {q}")
                    prompt = f"Dept={dept} Schema:{snippet}\nQ: {q}\nReturn JSON{{instruction,input,chain_of_thought,output}}"
                    raw = self._generate(prompt)
                    print("RAW:", raw)
                    transcript.append(f"RAW: {raw}")
                    rec = None
                    if '{' in raw and '}' in raw:
                        s,e = raw.find('{'), raw.rfind('}')
                        try: rec = json.loads(raw[s:e+1])
                        except: rec = None
                    if not rec:
                        rec = {'instruction': q, 'input': '', 'chain_of_thought': '', 'output': raw}
                    for k in ['instruction','input','chain_of_thought','output']:
                        rec.setdefault(k, '')
                    print("ANS:", rec)
                    transcript.append(f"ANS: {json.dumps(rec, ensure_ascii=False)}")
                    with open(DEFAULT_OUTPUT, 'a', encoding='utf-8') as out:
                        out.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    memory += f"Q:{q}\nA:{raw}\n"
                    for fq in self.generate_followups(q, raw, memory, dept, fups):
                        print("FUP:", fq)
                        transcript.append(f"FUP: {fq}")
                        if fq not in next_qs:
                            next_qs.append(fq)
                qs = next_qs
        Path(TRANSCRIPT_FILE).write_text("\n".join(transcript), encoding='utf-8')
        print(f"Training data -> {DEFAULT_OUTPUT}\nTranscript -> {TRANSCRIPT_FILE}")

    def interactive(self):
        self.generate_training_data()
        print("\n🎉 Generation complete — enter SQL queries or 'exit'.")
        while True:
            cmd = input("❓ ").strip()
            if cmd.lower() in ('exit','quit'):
                break
            self.ask(cmd)

    def ask(self, question: str) -> QueryResult:
        emb = self.embedder.encode([question])
        docs = self.collection.query(query_embeddings=emb.tolist(), n_results=5)['documents']
        ctx = docs[0] if docs else ''
        prompt = f"SQL only. Context:{ctx}\nQ:{question}"
        sql = self._generate(prompt)
        print("SQL:", sql)
        try:
            df = pd.read_sql(sql, pyodbc.connect(self.conn_str))
            print(df)
            return QueryResult(question, '', sql, df, '', True)
        except Exception as e:
            print(f"SQL error: {e}")
            return QueryResult(question, '', sql, pd.DataFrame(), str(e), False)

if __name__ == '__main__':
    RagDB().interactive()
