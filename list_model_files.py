from huggingface_hub import list_repo_files
for model_id in ["ai-forever/rugpt3large_based_on_gpt2", "ai-forever/rugpt3medium_based_on_gpt2"]:
    print(f"\n=== {model_id} ===")
    for f in list_repo_files(model_id):
        print(" ", f)
