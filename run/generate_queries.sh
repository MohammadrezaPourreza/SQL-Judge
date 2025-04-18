source .env

dataset_path="data/bird/train/train.json"
db_path="data/bird/train/train_databases/train_databases"
model_name="gemini-1.5-pro-002"
number_of_workers=10
number_of_samples=10

python3 -u ./src/generate_queries.py \
    --dataset_path $dataset_path \
    --db_path $db_path \
    --model_name $model_name \
    --number_of_workers $number_of_workers \
    --number_of_samples $number_of_samples
