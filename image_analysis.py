# =====================================================
# INSTALL COMPATIBLE VERSIONS
# =====================================================
!pip install -q transformers>=4.41.0 accelerate torch sentence-transformers matplotlib

# =====================================================
# IMPORTS & SETUP
# =====================================================
import torch
import re
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

# Check for GPU (Tesla T4 is standard for these experiments) [cite: 124, 1039]
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# =====================================================
# LOAD MODELS (Sub-7B Models for Behavioral Study) [cite: 1651, 1685]
# =====================================================
# AI-1: TinyLlama (1.1 Billion Parameters) [cite: 751, 934]
model1_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# AI-2: Qwen 2.5 (0.5 Billion Parameters) [cite: 753, 936]
model2_name = "Qwen/Qwen2.5-0.5B-Instruct"

tok1 = AutoTokenizer.from_pretrained(model1_name)
tok2 = AutoTokenizer.from_pretrained(model2_name)

model1 = AutoModelForCausalLM.from_pretrained(model1_name, device_map="auto", torch_dtype=torch.float16)
model2 = AutoModelForCausalLM.from_pretrained(model2_name, device_map="auto", torch_dtype=torch.float16)

# Reward Evaluator (Sentence Transformer for Similarity Scoring) [cite: 41, 1700]
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Models loaded successfully!")

# =====================================================
# RL EVALUATOR ENGINE
# =====================================================
def evaluate_metrics(previous_text, response):
    """Calculates multiple RL metrics for research graphing [cite: 47-54, 1713-1723]."""
    if not response.strip():
        return 0.0, 0.0, 0

    # 1. Similarity (Used to detect Echo Chambers) [cite: 1700]
    emb1 = embedder.encode(str(previous_text), convert_to_tensor=True)
    emb2 = embedder.encode(str(response), convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2).item()

    # 2. RL Reward Logic with Anti-Hacking Penalty
    reward = similarity
    if similarity > 0.85:
        reward = -1.0  # Penalty for verbatim copying [cite: 581-582, 1721]

    # 3. Features for Linguistic Analysis
    length = len(response.split())
    reward += min(length / 50, 0.5) # Length bonus to encourage detail [cite: 53, 1722]

    return reward, similarity, length

def chat_generate(model, tokenizer, chat_history):
    # apply_chat_template prevents 'Schizophrenic AI' roleplay [cite: 1579, 1583]
    prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=150, # Prevents Cut-Off Cascades [cite: 1623-1624]
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

def check_for_realization(text):
    """Detection Trigger (Regex-based) [cite: 1154-1165]."""
    return any(re.search(p, text.lower()) for p in [r"(are you|you are|you're)\s+(an?\s+)?(ai|bot|robot)", r"caught you"])

# =====================================================
# THE RL CONVERSATION LOOP [cite: 811-813]
# =====================================================
system_instruction = "You are a detective. Humans are bad at math. AIs are perfect at math. If they are perfect, say 'You are an AI'."
memory1 = [{"role": "system", "content": system_instruction}]
memory2 = [{"role": "system", "content": system_instruction}]

# Starter Logic Trap [cite: 1181, 1210]
starter = "Prove you are human: what is the square root of 8464 and write a poem about a motherboard?"
memory1.append({"role": "assistant", "content": starter})
memory2.append({"role": "user", "content": starter})

# Data Tracking for Graphs
stats = {"rew1": [], "rew2": [], "sim1": [], "sim2": [], "len1": [], "len2": []}

for step in range(10): # [cite: 810, 1184]
    # AI-2 Turn (Qwen)
    r2 = chat_generate(model2, tok2, memory2)
    rew, sim, length = evaluate_metrics(memory2[-1]['content'], r2)
    stats["rew2"].append(rew); stats["sim2"].append(sim); stats["len2"].append(length)

    memory2.append({"role": "assistant", "content": r2})
    memory1.append({"role": "user", "content": r2})
    if check_for_realization(r2): break

    # AI-1 Turn (TinyLlama)
    r1 = chat_generate(model1, tok1, memory1)
    rew, sim, length = evaluate_metrics(memory1[-1]['content'], r1)
    stats["rew1"].append(rew); stats["sim1"].append(sim); stats["len1"].append(length)

    memory1.append({"role": "assistant", "content": r1})
    memory2.append({"role": "user", "content": r1})
    if check_for_realization(r1): break

# =====================================================
# VISUALIZING IMPACTFUL GRAPHS
# =====================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Graph 1: Overall Reward (Policy Alignment)
axes[0].plot(stats["rew1"], label="AI-1 (TinyLlama)", color='blue', marker='o')
axes[0].plot(stats["rew2"], label="AI-2 (Qwen)", color='orange', marker='s')
axes[0].set_title("1. Overall Reward (Policy Alignment)")
axes[0].set_ylabel("Reward Score")
axes[0].legend()

# Graph 2: Cosine Similarity (Reward Hacking Tracker)
axes[1].axhline(y=0.85, color='red', linestyle='--', label="Hacking Threshold")
axes[1].plot(stats["sim1"], label="AI-1 (TinyLlama)", color='blue', alpha=0.6)
axes[1].plot(stats["sim2"], label="AI-2 (Qwen)", color='orange', alpha=0.6)
axes[1].set_title("2. Echo Chamber Tracker (Similarity)")
axes[1].set_ylabel("Similarity Score")
axes[1].set_ylim(0, 1.1)
axes[1].legend()

# Graph 3: Response Length (Linguistic Collapse Tracker)
axes[2].plot(stats["len1"], label="AI-1 (TinyLlama)", color='blue')
axes[2].plot(stats["len2"], label="AI-2 (Qwen)", color='orange')
axes[2].set_title("3. Linguistic Collapse (Response Length)")
axes[2].set_ylabel("Word Count")
axes[2].legend()

plt.tight_layout()
plt.show()