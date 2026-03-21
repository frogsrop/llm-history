from huggingface_hub import list_models
print("sberbank-ai:")
for m in list_models(author="sberbank-ai", search="rugpt3", limit=10):
    print(" ", m.id)
print("ai-forever:")
for m in list_models(author="ai-forever", search="rugpt", limit=10):
    print(" ", m.id)
