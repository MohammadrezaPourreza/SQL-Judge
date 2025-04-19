# from unsloth import FastLanguageModel, PatchFastRL
# from unsloth import is_bfloat16_supported
from src.database_utils.database_manager import get_db_schema_db_id, schema_linking_scorer
from src.database_utils.execution import compare_sqls, execute_sql
from src.prompts.prompt_loader import load_prompt
from datasets import load_dataset, DatasetDict 
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, TaskType, PeftModel
from src.llm_utils.llm import load_model, load_tokenizer
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
from typing import Any
from tqdm import tqdm

import concurrent.futures
import argparse
import os
import ast
import torch
import pandas as pd
import re
import wandb
import json

NAME = "sql-judge-7b-acc"

wandb.init(project="grpo-sql-judge-training-7B-model", name=NAME)

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
        """
        Formats the SQL query to pass to the picker model.
        
        Args:
            sql_meta_info (SQLMetaInfo): The SQL meta information.
        """
        try:
            db_path = os.path.join(db_path, db_id, f"{db_id}.sqlite")
            execution_result = execute_sql(
                db_path=db_path,
                sql=query,
            )
            number_of_rows = len(execution_result)
            if number_of_rows == 0:
                number_of_columns = 0
            else:
                number_of_columns = len(execution_result[0])
            if number_of_rows > 20:
                execution_result = execution_result[:20]
            formatted_result = (
                f"Rows: {number_of_rows}, Columns: {number_of_columns}, Results:"
                f" {execution_result}"
            )
            if len(formatted_result) > 2000:
                formatted_result = formatted_result[:2000]
        except Exception as e:
            formatted_result = "Error: " + str(e)
        return formatted_result

def process_row(row):
    """
    Process exactly one row of your DataFrame into the dict you append to training_datasets.
    Returns None on any exception, so we can filter failures out.
    """
    try:
        question        = row["question"]
        db_id           = row["db_id"]
        gold_query      = row["gold_query"]
        generated_query = row["generated_query"]
        evidence        = row["evidence"]
        label           = row["label"]

        # expensive I/O / CPU work
        schema = get_db_schema_db_id(
            db_id=db_id,
            bird_database_path=os.getenv("BASE_TRAIN_DATA_PATH"),
            queries=[gold_query, generated_query],
        )
        results = _format_sql_query_result(db_id, os.getenv("BASE_TRAIN_DATA_PATH"), generated_query)

        user_messages = RAW_PROMPT.format(
            QUESTION=question,
            DATABASE_SCHEMA=schema,
            HINT=evidence,
            SQL=generated_query,
            RESULTS=results,
        )

        return {
            'prompt': [
                {"role": "system", 'content': SYSTEM_PROMPT},
                {"role": "user",   'content': user_messages},
            ],
            'answer':  label,
            'db_id':    db_id,
            'question': question,
            'evidence': evidence,
            'gold_query': gold_query,
        }

    except Exception as e:
        # you could log row index + e here
        print(f"Error processing row: {e}")
        return None

def construct_finetuning_dataset():
    dataset_name = "finetuning_datasets/pointwise_judge_acc.csv"
    if os.path.exists(dataset_name):
        ds = load_dataset('csv', data_files=dataset_name)
        return ds.map(lambda x: {"prompt": json.loads(x["prompt"])})

    df = pd.read_json("results/sample_llm_responses_bird.json")
    df = df.sample(frac=0.25, random_state=42)

    os.makedirs("finetuning_datasets", exist_ok=True)
    training_datasets = []

    # choose ProcessPoolExecutor if CPU‐bound, ThreadPoolExecutor if I/O‐bound
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as exe:
        futures = [exe.submit(process_row, row) for _, row in df.iterrows()]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Building examples"):
            example = fut.result()
            if example is not None:
                training_datasets.append(example)

    # shuffle and save
    out_df = pd.DataFrame(training_datasets).sample(frac=1).reset_index(drop=True)
    print(len(out_df))
    out_df["prompt"] = out_df["prompt"].apply(json.dumps)
    out_df.to_csv(dataset_name, index=False)

    ds = load_dataset('csv', data_files=dataset_name)
    return ds.map(lambda x: {"prompt": json.loads(x["prompt"])})


def extract_label(response: str) -> int:
    match = re.search(r"<label>\s*(\d)\s*</label>", response)
    if match:
        return int(match.group(1))
    return -1

###### --------------------- REWARD FUNCTIONS --------------------- ######

def acc_reward_func(prompts, completions, answer, db_id, **kwargs) -> list[float]:
    print(f"Sample Completions :\n{completions[0][0]['content']}")
    responses = [extract_label(completion[0]['content']) for completion in completions]

    def evaluate(response, answer):
        try:
            if response == -1:
                return 0.0
            elif response == int(answer):
                return 3.0
        except Exception:
            return 0.0
        return 0.0

    # Use ThreadPoolExecutor to process items in parallel.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        rewards = list(executor.map(evaluate, responses, answer))
    
    print(f"Rewards: {rewards}")
    return rewards

### formatting reward functions:

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Strict reward function that checks if the completion has an exact format."""
    pattern = r"^\s*<reasoning>\s*.*?\s*</reasoning>\s*<label>\s*.*?\s*</label>\s*$"
    matches = [re.fullmatch(pattern, r, re.DOTALL) for r in completions]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Soft reward function that checks if the completion loosely follows the format."""
    pattern = r"<reasoning>\s*.*?\s*</reasoning>\s*<label>\s*.*?\s*</label>"
    matches = [re.search(pattern, r, re.DOTALL) for r in completions]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<label>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</label>\n")[-1])*0.001
    if text.count("\n</label>") == 1:
        count += 0.125
        count -= (len(text.split("\n</label>")[-1]) - 1)*0.001
    label = extract_label(text)
    if label == -1:
        count -= 0.5
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    reward = [count_xml(c[0]['content']) for c in completions]
    print(f"XML Count Rewards: {reward}")
    return reward

def train_model(dataset: Any, args: argparse.Namespace, tokenizer: Any, model: Any):
    training_args = GRPOConfig(
        use_vllm = True, # use vLLM for fast inference!
        vllm_device='cuda:0',
        vllm_gpu_memory_utilization=0.5,
        vllm_max_model_len=args.max_seq_length,
        learning_rate = 5e-5,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        save_total_limit=1,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "constant_with_warmup",
        optim = "paged_adamw_8bit",
        logging_steps = 1,
        bf16 = True,
        fp16 = False,
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        temperature=args.temperature,
        num_generations = args.num_generations,
        max_prompt_length = args.max_prompt_length,
        max_completion_length = args.max_completion_length,
        num_train_epochs = args.epochs, # Set to 1 for a full training run
        # max_steps = 250,
        epsilon_high=0.28,
        beta=0.0,
        num_iterations = 4,
        save_steps = 250,
        max_grad_norm = 0.2,
        scale_rewards=False,
        report_to = "wandb", # Can use Weights & Biases
        output_dir = args.output_model_name
    )

    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        r=args.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
    )

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            acc_reward_func,
            xmlcount_reward_func,
        ],
        args = training_args,
        train_dataset = dataset["train"],
        # eval_dataset= dataset["validation"],
        # peft_config=peft_config
    )
    train_results = trainer.train()

    return trainer, train_results

def filter_samples_based_on_length(example: Any, max_seq_length: int, tokenizer: Any):
    # user_messages = example["prompt"]
    # messages = [
    #     {"role": "user", "content": user_messages},
    # ]
    messages = example["prompt"]
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)) <= max_seq_length

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    args.add_argument("--max_seq_length", type=int, default=2548)
    args.add_argument("--max_prompt_length", type=int, default=2000)
    args.add_argument("--max_completion_length", type=int, default=548)
    args.add_argument("--lora_rank", type=int, default=32)
    args.add_argument("--lora_alpha", type=int, default=16)
    args.add_argument("--per_device_train_batch_size", type=int, default=30)
    args.add_argument("--epochs", type=int, default=1)
    args.add_argument("--gradient_accumulation_steps", type=int, default=10)
    args.add_argument("--num_generations", type=int, default=6) # Decrease if out of memory
    args.add_argument("--hf_username", type=str, default="MrezaPRZ")
    args.add_argument("--output_model_name", type=str, default=f"qwen2.5-Coder-7B-Instruct-{NAME}")
    args.add_argument("--temperature", type=int, default=0.9)
    args = args.parse_args()
    new_model_name = f"{args.hf_username}/{args.output_model_name}"
    model = load_model(args.model_name, quantize=False)
    tokenizer = load_tokenizer(args.model_name)
    dataset = construct_finetuning_dataset()
    dataset = dataset['train'].train_test_split(test_size=0.01, shuffle=True)
    dataset = DatasetDict({
        'train': dataset['train'],
        'validation': dataset['test']
    })
    dataset = dataset.filter(filter_samples_based_on_length, fn_kwargs={'max_seq_length': args.max_prompt_length, 'tokenizer': tokenizer})
    print(f"No of samples: {dataset['train'].shape[0]}")
    trainer, train_results = train_model(dataset, args, tokenizer, model)
    metrics = train_results.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    trainer.save_model(args.output_model_name)

    if trainer.accelerator.is_main_process:
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(args.output_model_name)
        trainer.tokenizer.push_to_hub(new_model_name)