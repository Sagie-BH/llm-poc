import json
import logging
import torch
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
        # Database connection
        self.conn_str = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=localhost;"
            "DATABASE=AdventureWorksDW2022;"
            "Trusted_Connection=yes;"
        )
        print("🚀 Initializing RagDB with LLM, GPU support, and training-data generator...")

        # LLM device
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"Using {'GPU' if self.device >= 0 else 'CPU'} for LLM inference.")

        # Embeddings (GPU if available)
        embed_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"📥 Loading embeddings on {embed_device}...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=embed_device)

        # Setup LLM
        print("🧠 Loading LLM model with quantization...")
        self._setup_llm()

        # Vector store
        print("🗄️ Setting up Chroma vector store...")
        Path("vectordb").mkdir(exist_ok=True)
        client = chromadb.PersistentClient(path="vectordb")
        self.collection = client.get_or_create_collection("db_knowledge")

        # Extract schema and index
        self.db_schema: Dict[str, Any] = {}
        self._extract_database_knowledge()
        self._build_knowledge_base()

        print("✅ RagDB ready with LLM integration")

    def _setup_llm(self):
        preferred = "meta-llama/Llama-3-8b"
        fallback = "TheBloke/vicuna-7B-1.1-HF"
        loaded = False
        for model_name in (preferred, fallback):
            try:
                print(f"Attempting to load model '{model_name}'...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map='auto',
                    load_in_4bit=(model_name == preferred),
                    torch_dtype=torch.float16
                )
                print(f"Loaded '{model_name}' successfully.")
                loaded = True
                break
            except Exception as ex:
                print(f"⚠️ Could not load '{model_name}': {ex}")
        if not loaded:
            raise RuntimeError(
                "Failed to load any LLM model. Ensure you have access and necessary packages installed."
            )
        gen_conf = GenerationConfig(max_new_tokens=128, temperature=0.7)
        self.text_generator = pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.tokenizer,
            do_sample=True,
            generation_config=gen_conf
        )

    def _extract_database_knowledge(self):
        print("🔍 Extracting database schema...")
        conn = pyodbc.connect(self.conn_str)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE='BASE TABLE' AND TABLE_SCHEMA='dbo'
              AND (TABLE_NAME LIKE 'Dim%' OR TABLE_NAME LIKE 'Fact%')
            ORDER BY TABLE_NAME
            """
        )
        tables = [r[0] for r in cursor.fetchall()]
        for tbl in tables:
            cursor.execute(
                f"SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='{tbl}'"
            )
            cols = cursor.fetchall()
            cursor.execute(f"SELECT TOP 1 * FROM dbo.{tbl}")
            sample = cursor.fetchall()
            col_names = [d[0] for d in cursor.description]
            cursor.execute(f"SELECT COUNT(*) FROM dbo.{tbl}")
            count = cursor.fetchone()[0]
            self.db_schema[tbl] = {
                'columns': [(c[0], c[1], c[2]) for c in cols],
                'sample': [dict(zip(col_names, row)) for row in sample],
                'row_count': count
            }
            print(f"   📊 {tbl}: {len(cols)} cols, {count:,} rows")
        conn.close()

    def _build_knowledge_base(self):
        if self.collection.count() > 0:
            print(f"📚 Knowledge base already initialized ({self.collection.count()} items)")
            return
        docs, metas, ids = [], [], []
        for tbl, info in self.db_schema.items():
            desc = f"Table {tbl}: columns {[c[0] for c in info['columns']]}, {info['row_count']} rows"
            docs.append(desc)
            metas.append({'table': tbl})
            ids.append(f"tbl_{tbl}")
            if info['sample']:
                docs.append(f"Sample row for {tbl}: {info['sample'][0]}")
                metas.append({'table': tbl, 'type': 'sample'})
                ids.append(f"sample_{tbl}")
        print(f"🧠 Embedding and indexing {len(docs)} documents on GPU...")
        embeddings = self.embedder.encode(docs, batch_size=64, show_progress_bar=True)
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=docs,
            metadatas=metas,
            ids=ids
        )
        print("✅ Knowledge base built on GPU")

    def build_initial_questions(self) -> List[str]:
        return [
            "List all tables and their columns.",
            "Identify key relationships between tables.",
            "What data types are used?"
        ]

    def generate_followups(self, question: str, response: str, memory: str, count: int = FOLLOWUP_COUNT) -> List[str]:
        prompt = (
            "You are a schema expert. Do not repeat prior questions. "
            f"Based on the memory and the last Q&A, propose {count} follow-up questions.\n"
            f"Memory:\n{memory}\n"
            f"Q: {question}\nA: {response}\n"
            "List questions as a JSON array."
        )
        raw = self.text_generator(prompt)[0]['generated_text']
        try:
            arr = raw[raw.find('['): raw.rfind(']')+1]
            return json.loads(arr)
        except Exception:
            return [line.strip('- ') for line in raw.splitlines() if line.strip()][:count]

    def generate_training_data(self, output_file: str = DEFAULT_OUTPUT, max_rounds: int = MAX_ROUNDS):
        transcript: List[str] = []
        memory = ""
        questions = self.build_initial_questions()
        schema_snip = json.dumps(self.db_schema, default=str)
        print(f"\n🔖 Generating training data ({max_rounds} rounds, {FOLLOWUP_COUNT} follow-ups each)")
        for rnd in range(1, max_rounds+1):
            transcript.append(f"=== Training Round {rnd} ===")
            print(f"\n=== Training Round {rnd} ===")
            next_questions: List[str] = []
            for q in questions:
                transcript.append(f"Q: {q}")
                prompt = (
                    "Show your chain-of-thought, then the final answer.\n"
                    f"Schema (truncated): {schema_snip[:TRUNCATE_LEN]}\n"
                    f"Q: {q}\n"
                    "Finally output a JSON object with keys: 'instruction','input','output'."
                )
                print(prompt)
                resp = self.text_generator(prompt)[0]['generated_text']
                transcript.append("Chain-of-thought + Answer:")
                transcript.append(resp)
                print("Chain-of-thought + Answer:\n", resp)
                record = {'instruction': q, 'input': '', 'output': resp}
                with open(output_file, 'a') as f:
                    f.write(json.dumps(record) + "\n")
                memory += f"Q: {q}\nA: {resp}\n"
                for fq in self.generate_followups(q, resp, memory):
                    if fq not in next_questions:
                        next_questions.append(fq)
                        transcript.append(f"Proposed follow-up: {fq}")
            questions = next_questions
        with open(TRANSCRIPT_FILE, 'w') as t:
            t.write("\n".join(transcript))
        print(f"✅ Training data → {output_file}")
        print(f"✅ Transcript → {TRANSCRIPT_FILE}")

    def interactive(self):
        # Auto-generate at startup
        self.generate_training_data()
        print("🎉 Generation complete. Enter SQL queries or 'exit'.")
        while True:
            cmd = input("❓ ").strip()
            if cmd.lower() in ('exit','quit'):
                print("👋 Goodbye!")
                break
            self.ask(cmd)

    def ask(self, question: str) -> QueryResult:
        print(f"\n🤔 Question: {question}")
        q_emb = self.embedder.encode([question])
        res = self.collection.query(query_embeddings=q_emb.tolist(), n_results=5)
        ctx = res['documents'][0] if res['documents'] else ''
        prompt = (
            "You are a SQL assistant. Use the schema context and output SQL only.\n"
            f"Context: {ctx}\nQ: {question}\n"
        )
        print(prompt)
        sql = self.text_generator(prompt)[0]['generated_text'].strip()
        print("🔍 Generated SQL:\n", sql)
        try:
            df = pd.read_sql(sql, pyodbc.connect(self.conn_str))
            print(f"✅ Retrieved {len(df)} rows.")
            success = True
        except Exception as e:
            print(f"❌ SQL error: {e}")
            df = pd.DataFrame()
            success = False
        return QueryResult(question, '', sql, df, '', success)

if __name__ == '__main__':
    # Clean old files
    for fn in (DEFAULT_OUTPUT, TRANSCRIPT_FILE):
        try: Path(fn).unlink()
        except: pass
    rag = RagDB()
    rag.interactive()