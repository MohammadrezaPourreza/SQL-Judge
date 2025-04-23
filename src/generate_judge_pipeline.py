import argparse
import json
import time
import re
import os
import yaml
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from llms.llm_engines import call_model
from database_utils.database_manager import get_db_schema_db_id
from database_utils.execution import execute_sql, compare_sqls
from prompts.prompt_loader import load_prompt


def setup_logging(level_name):
    """Set up logging with the specified level."""
    level = getattr(logging, level_name)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def extract_sql_queries(text: str) -> str:
    pattern = r"```sql\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip()


def extract_judge_output(text: str) -> tuple:
    """
    Extract reasoning and label from judge response.
    
    Args:
        text: The judge response text
        
    Returns:
        tuple: (reasoning, label)
    """
    # Extract reasoning
    reasoning_pattern = r"<reasoning>(.*?)</reasoning>"
    reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    
    # Extract label
    label_pattern = r"<label>(.*?)</label>"
    label_match = re.search(label_pattern, text, re.DOTALL)
    label_text = label_match.group(1).strip() if label_match else ""
    
    # Try to convert label to integer if it's a number
    try:
        label = int(label_text)
    except (ValueError, TypeError):
        label = label_text
        
    return reasoning, label


def process_sample(sample, args, gen_prompt_tpl, judge_prompt_tpl, logger):
    db_id = sample["db_id"]
    question = sample["question"]
    gold_query = sample["SQL"]
    evidence = sample.get("evidence", "")

    logger.info(f"Processing sample with db_id: {db_id}")
    logger.debug(f"Question: {question}")

    # Load DB schema
    try:
        db_schema = get_db_schema_db_id(db_id=db_id, bird_database_path=args.db_path)
    except Exception as e:
        logger.error(f"Failed to load DB schema for {db_id}: {str(e)}")
        return {
            "question": question,
            "gold_query": gold_query,
            "db_id": db_id,
            "rounds": [],
            "error": f"Failed to load DB schema: {str(e)}"
        }

    round_data = []
    prev_query = None

    # Full path to sqlite file
    sqlite_path = f"{args.db_path}/{db_id}/{db_id}.sqlite"

    for round_idx in range(1, args.max_retries + 2):
        logger.debug(f"Starting round {round_idx} for db_id: {db_id}")
        
        try:
            if round_idx == 1:
                # Initial generation
                prompt = gen_prompt_tpl.format(
                    DATABASE_SCHEMA=db_schema,
                    QUESTION=question,
                    HINT=evidence,
                )
                logger.debug("Calling generator model for initial generation")
                try:
                    response = call_model(args.generator_model_name, prompt, args.generator_temperature)
                except Exception as e:
                    logger.error(f"Generator model error in round {round_idx} for db_id {db_id}: {str(e)}")
                    # Add error information to round data and continue to next sample
                    round_data.append({
                        "round": round_idx,
                        "error": f"Generator model error: {str(e)}",
                        "query": "",
                        "model_response": f"Error: {str(e)}",
                        "exec_details": ([], f"Model failed: {str(e)}"),
                        "comparison": {"exec_res": 0, "exec_err": f"Model failed: {str(e)}"},
                        "label": 0,
                    })
                    # Exit rounds loop
                    raise ValueError(f"Generator model error: {str(e)}")
            else:
                # Fixer generation
                # Use previous query and its execution result
                results, _ = round_data[-1]["exec_details"]
                result_str = json.dumps(results, default=str)
                # First, evaluate the query using the judge model
                judge_prompt = judge_prompt_tpl.format(
                    DATABASE_SCHEMA=db_schema,
                    QUESTION=question,
                    SQL=prev_query,
                    RESULTS=result_str,
                    HINT=evidence,
                )
                logger.debug("Calling judge model for evaluation")
                try:
                    judge_response = call_model(args.judge_model_name, judge_prompt, args.judge_temperature)
                    
                    # Extract reasoning and label from judge response
                    reasoning, judge_label = extract_judge_output(judge_response)
                    logger.info(f"Judge score: {judge_label}, threshold: {args.judge_threshold}")
                except Exception as e:
                    logger.error(f"Error in judge model for db_id {db_id} round {round_idx}: {str(e)}")
                    # Create default reasoning and score
                    reasoning = f"Error during judging: {str(e)}"
                    judge_label = 0
                
                # If the judge score is below threshold, regenerate query
                if isinstance(judge_label, int) and judge_label < args.judge_threshold:
                    logger.info(f"Score below threshold, regenerating query")
                    # Use the original generation prompt with feedback
                    prompt = gen_prompt_tpl.format(
                        DATABASE_SCHEMA=db_schema,
                        QUESTION=question,
                        HINT=f"{evidence}\n\nPrevious attempt: {prev_query}\n\nFeedback: {reasoning}",
                    )
                    try:
                        response = call_model(args.generator_model_name, prompt, args.generator_temperature)
                    except Exception as e:
                        logger.error(f"Generator model error in round {round_idx} for db_id {db_id}: {str(e)}")
                        # Add error information to round data and continue to next sample
                        round_data.append({
                            "round": round_idx,
                            "error": f"Generator model error: {str(e)}",
                            "query": prev_query,
                            "model_response": f"Error: {str(e)}",
                            "exec_details": ([], f"Model failed: {str(e)}"),
                            "comparison": {"exec_res": 0, "exec_err": f"Model failed: {str(e)}"},
                            "label": 0,
                        })
                        # Break out of rounds loop and move to next sample
                        break
                else:
                    # Query is good enough, no need to regenerate
                    logger.info(f"Query deemed sufficient, stopping iterations")
                    break

            # Extract SQL query
            query = extract_sql_queries(response)
            prev_query = query
            logger.debug(f"Generated query: {query}")

            # Execute SQL to get results for fixer (fetch sample rows)
            try:
                logger.debug(f"Executing query against database")
                exec_rows = execute_sql(sqlite_path, query, fetch=args.fetch_rows, timeout=args.exec_timeout)
                exec_err = None
            except Exception as e:
                logger.warning(f"Query execution error: {str(e)}")
                exec_rows = []
                exec_err = str(e)

            # Compare against gold to get correctness label
            try:
                logger.debug("Comparing with gold query")
                comp = compare_sqls(
                    db_directory_path=args.db_path,
                    db_id=db_id,
                    predicted_sql=query,
                    ground_truth_sql=gold_query,
                )
                label = comp.get("exec_res", 0)
                error_flag = comp.get("exec_err", "")
                logger.info(f"Comparison result: {label}")
            except Exception as e:
                logger.error(f"Error comparing queries for db_id {db_id} round {round_idx}: {str(e)}")
                comp = {"exec_res": 0, "exec_err": str(e)}
                label = 0
                error_flag = str(e)

            # Record this round
            round_data.append({
                "round": round_idx,
                "query": query,
                "model_response": response,
                "exec_details": (exec_rows, exec_err),
                "comparison": comp,
                "label": label,
            })

            # Determine whether to continue
            if round_idx > args.max_retries:
                logger.info(f"Reached maximum retries ({args.max_retries}), stopping")
                break
            # If fixer should decide not to revise: no new SQL code in response
            if round_idx >= 1:
                # Check if fixer response contained a code block
                if not re.search(r"```sql", response):
                    logger.info("No SQL code found in fixer response, stopping")
                    break
                    
        except Exception as e:
            logger.error(f"Error in round {round_idx} for db_id {db_id}: {str(e)}")
            # Add error information to round data
            round_data.append({
                "round": round_idx,
                "error": str(e),
                "query": prev_query if prev_query else "",
                "exec_details": ([], f"Round failed: {str(e)}"),
                "comparison": {"exec_res": 0, "exec_err": f"Round failed: {str(e)}"},
                "label": 0,
            })
            # Continue to next round if not last round
            if round_idx < args.max_retries + 1:
                continue
            break

    logger.info(f"Completed processing for db_id: {db_id}")
    return {
        "question": question,
        "gold_query": gold_query,
        "db_id": db_id,
        "rounds": round_data,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-to-SQL pipeline with iterative fixing.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset JSON file.")
    parser.add_argument("--db_path", type=str, required=True, help="Path to the root database directory.")
    parser.add_argument("--generator_model_name", type=str, required=True, help="LLM model for SQL generation.")
    parser.add_argument("--judge_model_name", type=str, required=True, help="LLM model for pointwise judge.")
    parser.add_argument("--generator_prompt_template", type=str, required=True, help="Prompt template for SQL generation.")
    parser.add_argument("--judge_prompt_template", type=str, required=True, help="Prompt template for pointwise judge.")
    parser.add_argument("--generator_temperature", type=float, required=True, help="Sampling temperature.")
    parser.add_argument("--judge_temperature", type=float, required=True, help="Sampling temperature.")
    parser.add_argument("--max_retries", type=int, required=True, help="Maximum number of fixer retries.")
    parser.add_argument("--judge_threshold", type=float, required=True, help="Threshold for revision.")
    parser.add_argument("--fetch_rows", type=int, required=True, help="Number of rows to fetch for fixer.")
    parser.add_argument("--exec_timeout", type=int, required=True, help="Timeout for SQL execution (s).")
    parser.add_argument("--workers", type=int, required=True, help="Number of parallel workers.")
    parser.add_argument("--output", type=str, required=True, help="Directory to save pipeline results.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level (default: INFO)")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info(f"Starting text-to-SQL pipeline with {args.generator_model_name} generator and {args.judge_model_name} judge")

    # Create timestamp directory inside output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, timestamp)
    
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arguments at the beginning of the run
    args_file = os.path.join(output_dir, "args.yaml")
    with open(args_file, "w") as f:
        # Convert args to dict, excluding non-serializable values
        args_dict = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, list, dict)) or v is None}
        yaml.dump(args_dict, f, default_flow_style=False)
    logger.info(f"Arguments saved to {args_file}")

    # Load and prepare
    logger.info(f"Loading dataset from {args.dataset_path}")
    df = pd.read_json(args.dataset_path)
    logger.info(f"Loaded {len(df)} samples")
    
    gen_prompt_tpl = load_prompt(args.generator_prompt_template)
    judge_prompt_tpl = load_prompt(args.judge_prompt_template)

    results = []
    start = time.time()

    if args.workers == 1:
        # Single-threaded with tqdm progress bar
        logger.info("Running in single-threaded mode")
        for _, sample in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
            result = process_sample(sample, args, gen_prompt_tpl, judge_prompt_tpl, logger)
            results.append(result)
    else:
        # Multi-threaded with tqdm progress bar
        logger.info(f"Running with {args.workers} parallel workers")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_sample, sample, args, gen_prompt_tpl, judge_prompt_tpl, logger) 
                     for _, sample in df.iterrows()]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
                results.append(future.result())

    duration = time.time() - start
    logger.info(f"Pipeline completed in {duration:.2f}s for {len(results)} samples.")

    # Save results
    output_file = os.path.join(output_dir, "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_file}")
