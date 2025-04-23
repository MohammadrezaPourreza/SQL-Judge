source .env

dataset_path="/Users/stalaei/Desktop/Projects/SQL-Judge/data/dev/test.json"
db_path="/Users/stalaei/Desktop/Text2SQL/bird_dataset/dev/dev_databases/"
generator_model_name="gpt-4o"
judge_model_name="gpt-4o"
generator_prompt_template="sql_generation_zero_shot"
judge_prompt_template="pointwise_judge"
generator_temperature=0.7
judge_temperature=0.1
judge_threshold=7
max_retries=2
fetch_rows=20
exec_timeout=60
workers=16
output="results"
log_level="INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

python3 -u ./src/generate_judge_pipeline.py \
    --dataset_path $dataset_path \
    --db_path $db_path \
    --generator_model_name $generator_model_name \
    --judge_model_name $judge_model_name \
    --generator_prompt_template $generator_prompt_template \
    --judge_prompt_template $judge_prompt_template \
    --generator_temperature $generator_temperature \
    --judge_temperature $judge_temperature \
    --judge_threshold $judge_threshold \
    --max_retries $max_retries \
    --fetch_rows $fetch_rows \
    --exec_timeout $exec_timeout \
    --workers $workers \
    --output $output \
    --log_level $log_level
