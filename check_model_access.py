from huggingface_hub import model_info
try:
    info = model_info("ai-forever/rugpt3xl")
    print("Model found:", info.id)
    print("Private:", info.private)
    print("Gated:", getattr(info, 'gated', 'n/a'))
    print("Tags:", info.tags[:5] if info.tags else [])
except Exception as e:
    print("Error:", e)
