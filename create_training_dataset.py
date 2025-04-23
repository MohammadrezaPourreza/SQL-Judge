from src.database_utils.database_manager import get_db_schema_db_id, schema_linking_scorer
from src.database_utils.ngrams import jaccard_similarity
from src.database_utils.execution import compare_sqls, execute_sql
from src.prompts.prompt_loader import load_prompt
from datasets import load_dataset, DatasetDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dotenv import load_dotenv
from tqdm import tqdm

import os
import glob
import pandas as pd
import json

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<label>
...
</label>
"""
RAW_PROMPT = load_prompt("pointwise_judge")

load_dotenv(override=True)

def _format_sql_query_result(db_id, db_path, query) -> str:
    try:
        db_path = os.path.join(db_path, db_id, f"{db_id}.sqlite")
        execution_result = execute_sql(db_path=db_path, sql=query)
        number_of_rows = len(execution_result)
        number_of_columns = len(execution_result[0]) if number_of_rows > 0 else 0
        if number_of_rows > 20:
            execution_result = execution_result[:20]
        formatted_result = f"Rows: {number_of_rows}, Columns: {number_of_columns}, Results: {execution_result}"
        return formatted_result[:2000]
    except Exception as e:
        return "Error: " + str(e)

def syntax_check_scorer(
    db_id: str,
    gold_query: str,
):
    db_path  = os.getenv("BASE_TRAIN_DATA_PATH") + f"/{db_id}/{db_id}.sqlite"
    try:
        execute_sql(db_path=db_path, sql=gold_query, fetch="one")
        return 1.0
    except Exception as e:
        return 0.0

def process_record(record):
    try:
        question = record["question"]
        db_id = record["db_id"]
        gold_query = record["gold_query"]
        generated_query = record["generated_query"]
        evidence = record["evidence"]
        label = record["label"]

        schema = get_db_schema_db_id(
            db_id=db_id,
            bird_database_path=os.getenv("BASE_TRAIN_DATA_PATH"),
            queries=[generated_query],
        )
        results = _format_sql_query_result(db_id, os.getenv("BASE_TRAIN_DATA_PATH"), generated_query)
        ngram_jaccard = jaccard_similarity(generated_query, gold_query, n=2)
        schema_linking = schema_linking_scorer(gold_query, generated_query)
        syntax_check = syntax_check_scorer(db_id, generated_query)

        # user_messages = RAW_PROMPT.format(
        #     QUESTION=question,
        #     DATABASE_SCHEMA=schema,
        #     HINT=evidence,
        #     SQL=generated_query,
        #     RESULTS=results,
        # )

        return {
            'answer':  label,
            'db_id':    db_id,
            'question': question,
            'evidence': evidence,
            'gold_query': gold_query,
            'generated_query': generated_query,
            'schema': schema,
            'syntax_check': syntax_check,
            'schema_linking': schema_linking,
            'ngram_jaccard': ngram_jaccard,
            'formatted_result': results,
        }
    except Exception as e:
        print(f"Error processing record: {e}")
        return None

def construct_finetuning_dataset():
    # Output directory
    output_dir = "finetuning_datasets"
    os.makedirs(output_dir, exist_ok=True)

    # Load and sample data
    df = pd.read_json("results/sample_llm_responses_bird.json")
    records = df.to_dict('records')

    # Process in chunks of 5000 and dump CSV for each
    chunk_size = 2000
    for i in range(0, len(records), chunk_size):
        chunk = records[i : i + chunk_size]
        examples = []
        with ThreadPoolExecutor(max_workers=min(12, os.cpu_count())) as exe:
            for example in tqdm(exe.map(process_record, chunk), total=len(chunk), desc=f"Chunk {i//chunk_size}"):
                if example:
                    examples.append(example)

        out_df = pd.DataFrame(examples).sample(frac=1).reset_index(drop=True)
        chunk_file = os.path.join(output_dir, f"pointwise_judge_acc_{i//chunk_size}.csv")
        out_df.to_csv(chunk_file, index=False)
        print(f"Wrote {len(out_df)} examples to {chunk_file}")

    # Aggregate chunk files into one
    all_files = sorted(glob.glob(os.path.join(output_dir, "pointwise_judge_acc_*.csv")))
    df_list = [pd.read_csv(f) for f in all_files]
    big_df = pd.concat(df_list, ignore_index=True)
    big_file = os.path.join(output_dir, "queries_dataset.csv")
    big_df.to_csv(big_file, index=False)
    print(f"Aggregated {len(big_df)} examples into {big_file}")

    # Load as HuggingFace dataset
    ds = load_dataset('csv', data_files=big_file)
    return ds

if __name__ == "__main__":
    construct_finetuning_dataset()
