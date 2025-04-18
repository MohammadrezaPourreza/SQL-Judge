import json
import os
import pandas as pd
import re

from tqdm import tqdm
from collections import Counter


datasets_parent_path = "results"
filtered_dataset = "data/bird_train_filtered_train_filtered.csv"

sample_template = {
    "question": "",
    "answers": [],
    "generated_queries": [],
    "labels": [],
    "gold_query": "",
    "db_id": "",
    "evidence": ""
}

import re

try:
    import sqlparse
    SQLPARSE_AVAILABLE = True
except ImportError:
    SQLPARSE_AVAILABLE = False

def normalize_sql(query: str, use_sqlparse: bool = SQLPARSE_AVAILABLE) -> str:
    """
    Normalize a SQL query string by:
      - Removing comments (both inline '--' and block '/* */')
      - Converting to lower case
      - Removing extra whitespace, newlines, and tabs
      - Removing trailing semicolons
      - Cleaning up spaces around commas and parentheses
    Optionally uses sqlparse for formatting if installed.
    """
    if use_sqlparse:
        # Use sqlparse to strip comments and format the SQL.
        # keyword_case='lower' converts keywords to lowercase.
        formatted = sqlparse.format(query, keyword_case='lower', strip_comments=True)
    else:
        # Use regex based transformations if sqlparse is not available.
        formatted = query
        # Remove single-line comments (everything after --)
        formatted = re.sub(r'--.*', '', formatted)
        # Remove multi-line comments (/* ... */)
        formatted = re.sub(r'/\*.*?\*/', '', formatted, flags=re.DOTALL)
        # Convert to lowercase
        formatted = formatted.lower()
    
    # Remove newlines and tabs by replacing them with a single space.
    formatted = re.sub(r'[\n\t]+', ' ', formatted)
    # Collapse multiple spaces to a single space.
    formatted = re.sub(r'\s+', ' ', formatted)
    formatted = formatted.strip()
    # Remove any trailing semicolon.
    formatted = formatted.rstrip(';')
    # Additional cleanup: remove spaces around commas and parentheses.
    formatted = re.sub(r'\s*,\s*', ',', formatted)
    formatted = re.sub(r'\s*\(\s*', '(', formatted)
    formatted = re.sub(r'\s*\)\s*', ')', formatted)
    return formatted

def load_filtered_dataset(dataset_path):
    """
    Load the filtered dataset from a CSV file.
    """
    df = pd.read_csv(dataset_path)
    questions_to_filtered = set()
    for _, row in df.iterrows():
        question = row["question"]
        if row["overall_confidence"] == 10:
            questions_to_filtered.add(question)
    return questions_to_filtered


def load_all_json_files_in_directory(directory_path):
    """
    Load all JSON files in a directory and return their contents as a list of dictionaries.
    """
    data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            model_name = filename.split("_train_results.json")[0].split("_")[-1]
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r") as file:
                data.append({"data": json.load(file), "model_name": model_name})
    return data


def aggregate_all_results_based_on_question(all_data, questions_to_filtered):
    """
    Aggregate all results based on the question.
    """
    aggregated_results = []
    for result in all_data:
        for sample in result['data']:
            question = sample["question"]
            if question in questions_to_filtered:
                continue
            existing_sample = next((item for item in aggregated_results if item["question"] == question), None)
            if existing_sample:
                existing_sample["answers"].extend(sample["answers"])
                existing_sample["generated_queries"].extend(sample["generated_queries"])
                existing_sample["source_model"].extend([result["model_name"]] * len(sample["generated_queries"]))
                existing_sample["labels"].extend(sample["labels"])
            else:
                new_sample = sample_template.copy()
                new_sample.update(sample)
                new_sample["source_model"] = [result["model_name"]] * len(sample["generated_queries"])
                aggregated_results.append(new_sample)
    return aggregated_results

def remove_redundant_samples(aggregated_results):
    """
    Remove redundant samples from the aggregated results.
    """
    results_deduped = []
    for sample in tqdm(aggregated_results, desc="Removing Redundant Samples"):
        question = sample["question"]
        unique_generated_queries = []
        unique_answers = []
        unique_labels = []
        unique_models = []
        for generated_query, answer, label, model in zip(sample["generated_queries"], sample["answers"], sample["labels"], sample["source_model"]):
            normalized_query = normalize_sql(generated_query)
            if normalized_query not in unique_generated_queries:
                unique_generated_queries.append(normalized_query)
                unique_answers.append(answer)
                unique_labels.append(label)
                unique_models.append(model)
        sample["generated_queries"] = unique_generated_queries
        sample["answers"] = unique_answers
        sample["labels"] = unique_labels
        sample["source_model"] = unique_models
        results_deduped.append(sample)
    return results_deduped

if __name__ == "__main__":
    all_data = load_all_json_files_in_directory(datasets_parent_path)
    questions_to_filtered = load_filtered_dataset(filtered_dataset)
    aggregated_results = aggregate_all_results_based_on_question(all_data,questions_to_filtered)
    results_deduped = remove_redundant_samples(aggregated_results)

    final_dataset = []
    for sample in results_deduped:
        question = sample["question"]
        gold_query = sample["gold_query"]
        db_id = sample["db_id"]
        evidence = sample["evidence"]
        for generated_query, answer, label, model in zip(sample["generated_queries"], sample["answers"], sample["labels"], sample["source_model"]):
            final_dataset.append({
                "question": question,
                "gold_query": gold_query,
                "db_id": db_id,
                "evidence": evidence,
                "generated_query": generated_query,
                "model_answer": answer,
                "label": label,
                "source_model": model
            })
    # Save the final dataset to a JSON file
    df = pd.DataFrame(final_dataset)
    df.to_json("results/sample_llm_responses_bird.json", orient="records", indent=4)
    print(f"Final dataset size: {len(df)}")

    # show some statistics
    print("Number of queries per model:")
    print(Counter(df["source_model"].values.tolist()))
    
    print("Number of labels")
    print(Counter(df["label"].values.tolist()))