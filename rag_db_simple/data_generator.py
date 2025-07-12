# data_generator.py
# Enhanced generator with persistent memory, GPU utilization, and smarter follow-ups
# Now using Llama 3.1 (8B) for richer capabilities on RTX 4070 Super Ti

import json
import logging
import torch
from rag_db import RagDB
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# === Configuration ===
SCHEMA_DUMP = 'schema_for_training.json'
QA_ROUNDS_FILE = 'schema_qa_rounds.jsonl'
MODEL_NAME = 'meta-llama/Llama-3-8b'
MAX_ROUNDS = 3      # iterative QA layers
FOLLOWUP_COUNT = 3  # number of follow-up questions each time
MEMORY_LIMIT = 1000 # max characters of memory to include

# Automatically choose device: GPU if available
DEVICE = 0 if torch.cuda.is_available() else -1
if DEVICE >= 0:
    logging.info(f"CUDA is available. Using GPU device {DEVICE}.")
else:
    logging.info("CUDA not available. Falling back to CPU.")


def build_initial_questions() -> list[str]:
    return [
        "List all tables and their columns in the schema.",
        "Summarize table relationships in the schema.",
        "What data types are used across tables?"
    ]


def generate_followups_with_llm(generator, question: str, response: str, memory: str) -> list[str]:
    """
    Use LLM to propose follow-up questions, given Q&A and memory of past dialogue.
    Prevent repeats and focus on new details.
    """
    prompt = (
        "You are an expert on database schemas. Do not repeat previous questions."
        " Based on the memory of past Q&A and the latest Q&A below, propose"
        f" {FOLLOWUP_COUNT} concise, distinct follow-up questions that explore new aspects."
        "\nMemory of past dialogue:\n" + memory +
        "\n---\n"
        f"Question: {question}\n"
        f"Response: {response}\n"
        "---\n"
        "List the follow-up questions as a JSON array named 'questions'."
    )
    raw = generator(
        prompt,
        max_new_tokens=256,
        do_sample=False
    )[0]['generated_text']
    try:
        arr = raw[raw.find('['): raw.rfind(']')+1]
        return json.loads(arr)
    except Exception:
        return [line.strip('- ').strip() for line in raw.splitlines() if line.strip()][:FOLLOWUP_COUNT]


def query_and_log(generator, schema_snippet: str, question: str, memory: str, round_idx: int) -> dict:
    """
    Ask the model a question, including memory and schema snippet, and log the result.
    """
    prompt = (
        f"Round {round_idx}: You are a database documentation assistant. Provide chain-of-thought, then answer."  
        f"\nMemory (truncated):\n{memory}\n"
        f"Schema (truncated):\n{schema_snippet[:2000]}\n"
        f"Question: {question}\n"
        "Answer step by step, then the final result."
    )
    logging.info(f"[Round {round_idx}] Asking: {question}")
    generation_config = GenerationConfig(max_new_tokens=512, temperature=0)
    output = generator(
        prompt,
        generation_config=generation_config
    )[0]['generated_text']
    return {
        'round': round_idx,
        'question': question,
        'response': output,
        'prompt': prompt
    }


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    rag = RagDB()
    # Dump schema
    with open(SCHEMA_DUMP, 'w') as f:
        json.dump(rag.db_schema, f, indent=2, default=str)
    schema_snippet = json.dumps(rag.db_schema, default=str)

    # Load tokenizer & causal model with 4-bit quantization for RTX 4070
    logging.info(f"Loading model {MODEL_NAME}... ")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map='auto',
        load_in_4bit=True,
        torch_dtype=torch.float16
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        max_length=1024,
        do_sample=False
    )
    logging.info("Llama 3.1 model loaded for iterative Q&A.")

    all_qas = []
    questions = build_initial_questions()
    memory = ""

    for round_idx in range(1, MAX_ROUNDS+1):
        logging.info(f"Starting round {round_idx} with {len(questions)} questions.")
        next_questions = []
        for q in questions:
            qa = query_and_log(generator, schema_snippet, q, memory, round_idx)
            all_qas.append(qa)
            entry = f"Q: {qa['question']} A: {qa['response']}\n"
            memory = (memory + entry)[-MEMORY_LIMIT:]
            followups = generate_followups_with_llm(generator, qa['question'], qa['response'], memory)
            for fq in followups:
                if fq not in next_questions and fq not in memory:
                    next_questions.append(fq)
        questions = next_questions

    # Save all Q&A
    with open(QA_ROUNDS_FILE, 'w') as out_f:
        for qa in all_qas:
            out_f.write(json.dumps(qa, default=str) + "\n")
    logging.info(f"All {len(all_qas)} Q&A recorded in {QA_ROUNDS_FILE}")

if __name__ == '__main__':
    main()
