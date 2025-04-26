from transformers import pipeline

# 加载生成式问答模型（如 T5）
generator = pipeline("text2text-generation", model="google/flan-t5-base")

def generate_answer(question):
    prompt = f"回答以下问题：{question}"
    answer = generator(prompt, max_length=50)
    return answer[0]["generated_text"]

# 示例
print(generate_answer("what is your name?"))