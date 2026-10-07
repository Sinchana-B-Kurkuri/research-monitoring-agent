import csv
import random
import os

topics = [
    ("Autonomous Agents", "This paper explores autonomous agents using LLMs for planning and tool use."),
    ("Reinforcement Learning", "A novel RL algorithm for continuous control spaces."),
    ("Computer Vision", "Improving image classification with deep convolutional networks."),
    ("Natural Language Processing", "Attention mechanisms in transformer architectures for translation."),
    ("Robotics", "Motion planning for bipedal robots using inverse kinematics."),
    ("Multi-Agent Systems", "Cooperative strategies in multi-agent reinforcement learning."),
    ("Quantum Computing", "Error correction codes for near-term quantum processors."),
    ("Biology", "Protein folding predictions using neural networks."),
]

def generate_data(num_papers=200):
    papers = []
    for i in range(num_papers):
        topic = random.choice(topics)
        title = f"{topic[0]} Study {i+1}: Advanced Methods"
        abstract = f"{topic[1]} We present new results on the efficiency and scalability of this approach."
        papers.append({'paper_id': f"P{i:03d}", 'title': title, 'abstract': abstract})
    
    os.makedirs('data', exist_ok=True)
    with open('data/sample_papers.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['paper_id', 'title', 'abstract'])
        writer.writeheader()
        writer.writerows(papers)

if __name__ == "__main__":
    generate_data()
    print("Generated 200 sample papers in data/sample_papers.csv")
