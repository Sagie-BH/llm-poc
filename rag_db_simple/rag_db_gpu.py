import torch
print("Torch version:       ", torch.__version__)
print("Built with CUDA?     ", torch.version.cuda is not None)
print("CUDA available?      ", torch.cuda.is_available())
print("CUDA devices count:  ", torch.cuda.device_count())

import json
import logging
import time
import pandas as pd
import pyodbc
import chromadb
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

# === Configurable parameters ===
MAX_ROUNDS = 3       # Number of training data generation iterations
FOLLOWUP_COUNT = 3   # Number of follow-up questions per iteration
DEFAULT_OUTPUT = 'training.jsonl'  # Default file for generated data
TRANSCRIPT_FILE = 'transcript.txt'  # Human-readable conversation log
TRUNCATE_LEN = 500   # Characters of schema to include in each prompt
MAX_TOKENS = 64      # Limit generation length for speed

# Departments to cycle through for follow-up perspectives
dEPARTMENTS = ["Sales", "HR", "Finance", "IT", "Marketing"]

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
        print("🚀 Initializing RagDB with LLM, GPU support, and training-data generator...")
        self.conn_str = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=localhost;"
            "DATABASE=AdventureWorksDW2022;"
            "Trusted_Connection=yes;"
        )

        # Device selection
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"Using {'GPU' if self.device >= 0 else 'CPU'} for LLM inference.")

        # Embeddings
        embed_dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"📥 Loading embeddings on {embed_dev}...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=embed_dev)

        # Load LLM
        print("🧠 Loading LLM model with quantization (if GPU)...")
        self._setup_llm()

        # Vector store
        print("🗄️ Setting up Chroma vector store...")
        Path("vectordb").mkdir(exist_ok=True)
        client = chromadb.PersistentClient(path="vectordb")
        self.collection = client.get_or_create_collection("db_knowledge")

        # Schema extraction
        self.db_schema: Dict[str, Any] = {}
        self._extract_database_knowledge()
        self._build_knowledge_base()

        print("✅ RagDB ready with LLM integration")

    def _setup_llm(self):
        preferred = "meta-llama/Llama-3-8b"
        fallback = "TheBloke/vicuna-7B-1.1-HF"
        # If no GPU, use a tiny model for CPU testing
        model_names = [preferred, fallback] if self.device >= 0 else ["gpt2"]

        loaded = False
        for name in model_names:
            try:
                print(f"Attempting to load model '{name}'...")
                self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False)
                model_kwargs: Dict[str, Any] = {}
                if self.device >= 0 and name == preferred:
                    model_kwargs.update({
                        "device_map": "auto",
                        "load_in_4bit": True,
                        "torch_dtype": torch.float16
                    })
                elif self.device >= 0:
                    model_kwargs.update({
                        "device_map": "auto",
                        "torch_dtype": torch.float16
                    })
                self.llm_model = AutoModelForCausalLM.from_pretrained(name, **model_kwargs)
                print(f"Loaded '{name}' successfully.")
                loaded = True
                break
            except Exception as e:
                print(f"⚠️ Could not load '{name}': {e}")

        if not loaded:
            raise RuntimeError("Failed to load any LLM model. Check access and prerequisites.")

        # Sampling config ensures chain-of-thought is captured
        gen_conf = GenerationConfig(max_new_tokens=MAX_TOKENS, temperature=0.7, do_sample=True)

        # Omit explicit 'device' when model is loaded via accelerate
        self.text_generator = pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            generation_config=gen_conf
        )

    def _extract_database_knowledge(self):
        print("🔍 Extracting database schema...")
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
        tables = [r[0] for r in cur.fetchall()]
        for t in tables:
            cur.execute(f"SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='{t}'")
            cols = cur.fetchall()
            cur.execute(f"SELECT TOP 1 * FROM dbo.{t}")
            sample = cur.fetchall()
            names = [d[0] for d in cur.description]
            cur.execute(f"SELECT COUNT(*) FROM dbo.{t}")
            cnt = cur.fetchone()[0]
            self.db_schema[t] = {
                'columns': [(c[0], c[1], c[2]) for c in cols],
                'sample': [dict(zip(names, row)) for row in sample],
                'row_count': cnt
            }
            print(f"   📊 {t}: {len(cols)} cols, {cnt:,} rows")
        conn.close()

    def _build_knowledge_base(self):
        if self.collection.count() > 0:
            print(f"📚 Knowledge base already initialized ({self.collection.count()} items)")
            return
        docs, metas, ids = [], [], []
        for t, info in self.db_schema.items():
            desc = f"Table {t}: columns {[c[0] for c in info['columns']]}, {info['row_count']} rows"
            docs.append(desc); metas.append({'table': t}); ids.append(f"tbl_{t}")
            if info['sample']:
                docs.append(f"Sample row for {t}: {info['sample'][0]}")
                metas.append({'table': t, 'type': 'sample'}); ids.append(f"sample_{t}")
        print(f"🧠 Embedding and indexing {len(docs)} docs...")
        embs = self.embedder.encode(docs, batch_size=64, show_progress_bar=True)
        self.collection.add(embeddings=embs.tolist(), documents=docs, metadatas=metas, ids=ids)
        print("✅ Knowledge base built")

    def _generate(self, prompt: str) -> str:
        print("🔄 Generating output…")
        start = time.time()
        resp = self.text_generator(prompt)[0]['generated_text']
        duration = time.time() - start
        print(f"🔔 Generation done in {duration:.1f}s")
        print(resp)
        return resp

    def build_initial_questions(self) -> List[str]:
        return [
            "List all tables and their columns.",
            "Identify key relationships between tables.",
            "What data types are used?"
        ]

    def generate_followups(self, q: str, a: str, mem: str, department: str, count: int = FOLLOWUP_COUNT) -> List[str]:
        p = (
            f"You are a schema expert and an employee in the {department} department. "
            "Do not repeat prior questions. Propose follow-up questions to clarify the previous answer for your department.\n"
            f"Memory:\n{mem}\nQ: {q}\nA: {a}\n"
            "Return a JSON array of strings."
        )
        out = self._generate(p)
        try:
            arr = out[out.find('['): out.rfind(']')+1]
            return json.loads(arr)
        except:
            lines = [l.strip('- ') for l in out.splitlines() if l.strip()]
            return lines[:count]

    def generate_training_data(self, output_file: str = DEFAULT_OUTPUT, max_rounds: int = MAX_ROUNDS):
        transcript = []
        mem = ""
        qs = self.build_initial_questions()
        schema_snip = json.dumps(self.db_schema, default=str)
        print(f"\n🔖 Generating training data ({max_rounds} rounds x {FOLLOWUP_COUNT} follow-ups)")

        for r in range(1, max_rounds+1):
            dept = dEPARTMENTS[(r-1) % len(dEPARTMENTS)]
            transcript.append(f"=== Round {r} ({dept} Dept) ===")
            print(f"\n=== Training Round {r} ({dept} Dept) ===")
            next_qs = []

            for q in qs:
                print(f"\n[Round {r}] Q: {q}")
                transcript.append(f"Q: {q}")
                prompt = (
                    f"You are a schema expert and an employee in the {dept} department. "
                    "Think step-by-step (chain-of-thought) and then answer the question."
                    f"Schema snippet: {schema_snip[:TRUNCATE_LEN]}\n"
                    f"Q: {q}\n"
                    "Return JSON with keys: instruction, input, chain_of_thought, output."
                )
                raw = self._generate(prompt)
                # attempt to parse JSON response
                try:
                    rec = json.loads(raw)
                except Exception:
                    rec = {
                        'instruction': q,
                        'input': '',
                        'chain_of_thought': '',
                        'output': raw
                    }
                # write training example
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                mem += f"Q: {q}\nA: {raw}\n"
                transcript.append(raw)

                # generate follow-ups
                fqs = self.generate_followups(q, raw, mem, dept)
                for fq in fqs:
                    if fq not in next_qs:
                        next_qs.append(fq)
                        print(f"Follow-up: {fq}")
                        transcript.append(f"Follow-up: {fq}")

            qs = next_qs

        # save transcript
        with open(TRANSCRIPT_FILE, 'w', encoding='utf-8') as t:
            t.write("\n".join(transcript))
        print(f"✅ Training data -> {output_file}")
        print(f"✅ Transcript -> {TRANSCRIPT_FILE}")

    def interactive(self):
        self.generate_training_data()
        print("🎉 Generation complete. Enter SQL queries or 'exit'.")
        while True:
            cmd = input("❓ ").strip()
            if cmd.lower() in ('exit', 'quit'):
                print("👋 Goodbye!")
                break
            self.ask(cmd)

    def ask(self, question: str) -> QueryResult:
        print(f"\n🤔 Question: {question}")
        emb = self.embedder.encode([question])
        res = self.collection.query(query_embeddings=emb.tolist(), n_results=5)
        ctx = res['documents'][0] if res['documents'] else ''
        prompt = (
            "You are a SQL assistant. Use only the schema context below to write a valid SQL query without any extra text.\n"
            f"Context: {ctx}\n"
            f"Q: {question}\n"
            "Return only the SQL statement; do not include any explanations or extra tokens."
        )
        print(prompt)
        sql = self._generate(prompt)
        try:
            df = pd.read_sql(sql, pyodbc.connect(self.conn_str))
            print(f"✅ Retrieved {len(df)} rows.")
            return QueryResult(question, '', sql, df, '', True)
        except Exception as e:
            print(f"❌ SQL error: {e}")
            return QueryResult(question, '', sql, pd.DataFrame(), '', False)

if __name__ == '__main__':
    for fn in (DEFAULT_OUTPUT, TRANSCRIPT_FILE):
        try:
            Path(fn).unlink()
        except:
            pass
    rag = RagDB()
    rag.interactive()
