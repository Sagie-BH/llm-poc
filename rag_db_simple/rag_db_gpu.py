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
MAX_TOKENS = 128
SCHEMA_SNIPPET_LEN = 2000
DEFAULT_OUTPUT = 'training.jsonl'
TRANSCRIPT_FILE = 'transcript.txt'

# Department-specific configuration
departments = [
    {'name': 'Sales',     'rounds': 3, 'followups': 3},
    {'name': 'HR',        'rounds': 3, 'followups': 3},
    {'name': 'Finance',   'rounds': 3, 'followups': 3},
    {'name': 'IT',        'rounds': 3, 'followups': 3},
    {'name': 'Marketing', 'rounds': 3, 'followups': 3}
]

dept_filters = {
    'Sales': ['Sales','InternetSales','ResellerSales'],
    'HR': ['Employee','Department'],
    'Finance': ['Finance','CurrencyRate'],
    'IT': ['CallCenter','SurveyResponse','Scenario'],
    'Marketing': ['Promotion','Product','SalesReason']
}

initial_questions = {
    'Sales': [
        "List all AdventureWorks sales tables and their columns.",
        "Which tables and foreign keys impact monthly revenue reporting?",
        "What data types are used for sales metrics fields?"
    ],
    'HR': [
        "List all AdventureWorks HR tables and their columns.",
        "Which tables model employee-manager hierarchies?",
        "What data types are used in HR analytics tables?"
    ],
    'Finance': [
        "List all AdventureWorks finance tables and their columns.",
        "Which tables and keys support budgeting and P&L reports?",
        "What data types support financial aggregates?"
    ],
    'IT': [
        "List all AdventureWorks IT/operations tables and their columns.",
        "Which tables map servers to applications?",
        "What data types store performance/log metrics?"
    ],
    'Marketing': [
        "List all AdventureWorks marketing tables and their columns.",
        "Which tables track campaign and channel performance?",
        "What data types support marketing KPIs?"
    ]
}

# Few-shot example formatter
def few_shot_example(dept: str) -> str:
    example = {
        "instruction": initial_questions[dept][0],
        "input": "",
        "chain_of_thought": f"Filter tables by {dept} keywords, then list columns.",
        "output": [
            "DimSalesTerritory: TerritoryKey, Name, Country, SalesYTD, SalesLastYear",
            "FactInternetSales: SalesOrderKey, OrderQuantity, SalesAmount"
        ]
    }
    return json.dumps(example)

@dataclass
class QueryResult:
    question: str
    sql_query: str
    results: pd.DataFrame
    error: str

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

        # Load embeddings
        dev = 'cuda' if self.device>=0 else 'cpu'
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=dev)

        # Load LLM
        self._setup_llm()

        # Build vector store
        Path('vectordb').mkdir(exist_ok=True)
        client = chromadb.PersistentClient(path='vectordb')
        self.collection = client.get_or_create_collection('db_knowledge')

        # Extract and embed schema
        self.db_schema: Dict[str,Any] = {}
        self._extract_schema()
        self._embed_schema()

    def _setup_llm(self):
        # Choose model list
        names = ['meta-llama/Llama-3-8b','TheBloke/vicuna-7B-1.1-HF'] if self.device>=0 else ['gpt2']
        for nm in names:
            try:
                logging.info(f"Loading LLM {nm}...")
                self.tokenizer = AutoTokenizer.from_pretrained(nm, use_fast=False)
                opts = {}
                if self.device>=0:
                    opts = {'device_map':'auto','torch_dtype':torch.float16}
                    if 'Llama-3' in nm: opts['load_in_4bit']=True
                self.llm_model = AutoModelForCausalLM.from_pretrained(nm, **opts)
                logging.info(f"Loaded {nm}")
                break
            except Exception as e:
                logging.warning(f"Failed to load {nm}: {e}")
        gen_conf = GenerationConfig(max_new_tokens=MAX_TOKENS, temperature=0.7, do_sample=True)
        self.text_generator = pipeline(
            'text-generation', model=self.llm_model, tokenizer=self.tokenizer,
            return_full_text=False, generation_config=gen_conf
        )

    def _extract_schema(self):
        conn=pyodbc.connect(self.conn_str); cur=conn.cursor()
        cur.execute(
            "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
            "WHERE TABLE_TYPE='BASE TABLE' AND TABLE_SCHEMA='dbo' "
            "AND (TABLE_NAME LIKE 'Dim%' OR TABLE_NAME LIKE 'Fact%')"
        )
        for (tbl,) in cur.fetchall():
            cols = cur.execute(
                f"SELECT COLUMN_NAME,DATA_TYPE,IS_NULLABLE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='{tbl}'"
            ).fetchall()
            sample = cur.execute(f"SELECT TOP 1 * FROM dbo.{tbl}").fetchall(); names = [d[0] for d in cur.description]
            cnt = cur.execute(f"SELECT COUNT(*) FROM dbo.{tbl}").fetchone()[0]
            self.db_schema[tbl] = {'columns':cols,'sample':[{n:v for n,v in zip(names,row)} for row in sample],'row_count':cnt}
            logging.info(f"Loaded schema {tbl}: {len(cols)} cols, {cnt} rows")
        conn.close()

    def _embed_schema(self):
        if self.collection.count()>0: return
        docs, metas, ids = [], [], []
        for tbl,info in self.db_schema.items():
            docs.append(f"{tbl}: columns {[c[0] for c in info['columns']]} rows {info['row_count']}")
            metas.append({'table':tbl}); ids.append(f"tbl_{tbl}")
            if info['sample']:
                docs.append(f"Sample {tbl}: {info['sample'][0]}")
                metas.append({'table':tbl,'type':'sample'}); ids.append(f"smp_{tbl}")
        embs = self.embedder.encode(docs, batch_size=64)
        self.collection.add(embeddings=embs.tolist(),documents=docs,metadatas=metas,ids=ids)
        logging.info("Schema embeddings added")

    def _prompt_with_context(self, dept:str, question:str, memory:str) -> str:
        # grab dept-specific schema
        keys = dept_filters.get(dept,[])
        snippet = {t:self.db_schema[t] for t in self.db_schema if any(k in t for k in keys)}
        schema_snip = json.dumps(snippet, default=str)[:SCHEMA_SNIPPET_LEN]
        example = few_shot_example(dept)
        return (
            f"Example:{example}\n"
            f"Dept:{dept} Schema:{schema_snip}\n"
            f"Memory:{memory}\nQ:{question}\n"
            f"Return JSON with keys instruction,input,chain_of_thought,output."
        )

    def _generate(self,prompt:str)->str:
        return self.text_generator(prompt)[0]['generated_text']

    def build_initial(self,dept:str)->List[str]:
        return initial_questions.get(dept,[])

    def generate_followups(self, q:str, a:str, mem:str, dept:str, cnt:int)->List[str]:
        # department expert follow-ups
        prompt = (
            f"You are a {dept} domain expert. "
            f"Based on conversation memory:\n{mem}\nQ:{q}\nA:{a}\n"
            f"Suggest {cnt} focused follow-up questions (ending with '?') as JSON array."
        )
        raw = self._generate(prompt)
        try:
            arr = json.loads(raw[raw.find('['):raw.rfind(']')+1])
        except:
            arr = [ln.strip().rstrip('.') for ln in raw.splitlines() if ln.strip()]
        out=[]
        for itm in arr:
            qs = itm if isinstance(itm,str) else json.dumps(itm)
            qs = qs.strip('" ')
            if qs.endswith('?') and 'pizza' not in qs.lower(): out.append(qs)
            if len(out)==cnt: break
        return out

    def generate_training_data(self):
        # reset files
        for f in (DEFAULT_OUTPUT,TRANSCRIPT_FILE):
            try: Path(f).unlink()
            except: pass
        transcript=[]
        # iterate each department
        for cfg in departments:
            dept, rounds, fup_cnt = cfg['name'], cfg['rounds'], cfg['followups']
            qs = self.build_initial(dept)
            mem = ''
            for r in range(1,rounds+1):
                print(f"=== Dept:{dept} Round:{r}/{rounds} ===")
                transcript.append(f"=== Dept:{dept} Round:{r}/{rounds} ===")
                next_qs=[]
                for q in qs:
                    prompt = self._prompt_with_context(dept,q,mem)
                    raw = self._generate(prompt)
                    print("RAW:",raw)
                    transcript.append(f"RAW:{raw}")
                    # parse JSON or fallback
                    try: rec = json.loads(raw)
                    except: rec = {'instruction':q,'input':'','chain_of_thought':'','output':raw}
                    # sanitize generic
                    if any(t in rec['instruction'].lower() for t in ['pizza','json']): continue
                    print("ANS:",rec)
                    transcript.append(f"ANS:{json.dumps(rec)}")
                    with open(DEFAULT_OUTPUT,'a') as fo: fo.write(json.dumps(rec)+'\n')
                    mem += f"Q:{q}\nA:{rec['output']}\n"
                    # generate follow-ups
                    for fq in self.generate_followups(q,rec['output'],mem,dept,fup_cnt):
                        print("FUP:",fq)
                        transcript.append(f"FUP:{fq}")
                        next_qs.append(fq)
                qs = next_qs
        Path(TRANSCRIPT_FILE).write_text("\n".join(transcript))
        print("Training data and transcript created.")

    def interactive(self):
        self.generate_training_data()
        print("Generation complete. Enter SQL or 'exit'.")
        while True:
            cmd = input("SQL> ").strip()
            if cmd.lower() in ('exit','quit'): break
            self.ask(cmd)

    def ask(self, question:str)->QueryResult:
        emb = self.embedder.encode([question])
        docs = self.collection.query(query_embeddings=emb.tolist(),n_results=5)['documents']
        ctx = docs[0] if docs else ''
        prompt = f"SQL only. Context:{ctx}\nQ:{question}"
        sql = self._generate(prompt)
        print("SQL:",sql)
        try:
            df = pd.read_sql(sql, pyodbc.connect(self.conn_str))
            print(df)
            return QueryResult(question, sql, df, '')
        except Exception as e:
            print(f"Error executing SQL: {e}")
            return QueryResult(question, sql, pd.DataFrame(), str(e))

if __name__ == '__main__':
    RagDB().interactive()
