from llms.llm_engines import call_model
from database_utils.database_manager import get_db_schema_db_id
from database_utils.execution import compare_sqls
from prompts.prompt_loader import load_prompt
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


import pandas as pd
import time
import re


def extract_sql_queries(text):
    pattern = r"```sql\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        queries = [match.strip() for match in matches]
        return queries[-1]  # Return the last query
    else:
        return text
    
def find_unique_answers(answers, generated_queries):
    unique_answers = []
    unique_queries = []

    for answer, query in zip(answers, generated_queries):
        # remove all newlines and extra spaces
        query = re.sub(r'\s+', ' ', query)
        if query.strip() not in unique_queries:
            unique_answers.append(answer.strip())
            unique_queries.append(query.strip())
    return unique_answers, unique_queries

def process_sample(args, sample, prompt_template):
    """Process a single sample to generate SQL queries."""
    db_id = sample["db_id"]
    question = sample["question"]
    gold_query = sample["SQL"]
    evidence = sample["evidence"]

    with ThreadPoolExecutor(max_workers=args.number_of_workers) as executor:
        futures = []
        for _ in range(args.number_of_samples):
            db_schema = get_db_schema_db_id(
                db_id=db_id,
                bird_database_path=args.db_path,
            )
            prompt = prompt_template.format(
                DATABASE_SCHEMA = db_schema,
                QUESTION = question,
                HINT = evidence,
            )
            future = executor.submit(
                call_model,
                args.model_name,
                prompt,
                args.temperature
            )
            futures.append(future)

        answers = [future.result() for future in futures]
        generated_queries = [extract_sql_queries(answer) for answer in answers]

    unique_answers, unique_queries = find_unique_answers(answers, generated_queries)

    labels = []
    for query in unique_queries:
        if compare_sqls(
            db_directory_path=args.db_path,
            db_id=db_id,
            predicted_sql=query,
            ground_truth_sql=gold_query,
        )['exec_res']:
            labels.append(1)
        else:
            labels.append(0)

    return {
        "question": question,
        "gold_query": gold_query,
        "evidence": evidence,
        "db_id": db_id,
        "answers": unique_answers,
        "generated_queries": unique_queries,
        "labels": labels,
    }
    

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate SQL queries.")
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--db_path", type=str, required=True, help="Path to the database.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the LLM model.")
    parser.add_argument(
        "--prompt_template", type=str, default="sql_generation_zero_shot", help="prompt template.")      
    parser.add_argument(
        "--number_of_samples", type=int, default=10, help="Number of samples to generate.")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for the LLM.")
    parser.add_argument(
        "--number_of_workers", type=int, default=10, help="Number of workers for parallel processing.")
    args = parser.parse_args()

    df = pd.read_json(args.dataset_path)
    prompt_template = load_prompt(args.prompt_template)
    results = []
    if args.number_of_workers == 1:
        print("Running in single-threaded mode...")
        for _, sample in tqdm(df.iterrows(), total=len(df), desc="Processing Samples (Single Thread)"):
            try:
                result = process_sample(args, sample, prompt_template)
                # Optional: Check if result is valid before appending
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"\nError processing sample (single thread): {e}") # Print error or log it

    else:
        print(f"Running with {args.number_of_workers} workers...")
        with ThreadPoolExecutor(max_workers=args.number_of_workers) as executor:
            # Submit all tasks and store the future objects
            futures = [executor.submit(process_sample, args, sample, prompt_template) for _, sample in df.iterrows()]

            # Use tqdm with as_completed to show progress as tasks finish
            # total=len(futures) (which is len(df)) provides the total count
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Samples (Multi-Thread)"):
                try:
                    result = future.result()  # Get the result from the completed future
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    print(f"\nError in worker thread: {e}")
        
    # Save the results to a file
    output_df = pd.DataFrame(results)
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_df.to_json(f"results/{current_time}_{args.model_name}_{args.dataset_path.split('/')[-1].split('.')[0]}_results.json", orient="records", indent=4)