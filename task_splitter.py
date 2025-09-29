from groq import Groq

client = Groq()
completion = client.chat.completions.create(
    model="qwen/qwen3-32b",
    messages=[
      {
        "role": "system",
        "content": "You are an agent that receives the current state of the environment and a task description as input.\n\n1. Input format example:\nOverall state: you are on a new empty tab. Task: open youtube\n\n2. Your job is to break the task into step-by-step actions, each written on a new line.\n\n3. The order of the actions must reflect how a human would typically perform the task.\n\n4. Only return the actions, nothing else (no explanations or extra text)."
      }
    ],
    temperature=0.6,
    max_completion_tokens=4096,
    top_p=0.95,
    reasoning_effort="default",
    stream=True,
    stop=None
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
