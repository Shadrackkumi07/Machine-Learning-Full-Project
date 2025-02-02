# 🤖 Machine Learning Full Project – AI vs. DOOM Boss 🚀  

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1AVR5mJ3N1BUV7__ElWe_mlkEXgVtZAkk" alt="Training AI Model GIF">
</p>


## 📝 Overview  

This project is part of my **final semester** and involves a team of **three members**. Our goal is to develop a **Machine Learning (ML) model** that can **train an AI agent to defeat a game boss in DOOM** 🎮 using **deep reinforcement learning (DRL).**  

We aim to implement a **self-learning AI** that continuously improves by interacting with the game environment, making real-time decisions, and developing advanced strategies to defeat the **DOOM boss**.  

---

## 🛠️ **Tech Stack & Tools**  

✅ **Game Environment** 🎮  
- **ViZDoom** – A reinforcement learning environment for training AI in DOOM  
- OpenAI Gym – For standardized RL training and evaluation  

✅ **Programming & Frameworks** 💻  
- **Python** – Primary language for AI training  
- **TensorFlow / PyTorch** – Deep learning frameworks for model development  
- OpenCV – Computer vision library for processing game frames  

✅ **Reinforcement Learning Algorithms** 🧠  
- **Deep Q-Learning (DQN)** – A neural network-based RL technique  
- **Proximal Policy Optimization (PPO)** – Policy-gradient method for training agents  
- **A3C (Asynchronous Advantage Actor-Critic)** – For parallel training  

✅ **Data Collection & Processing** 📊  
- **Frame Capture & Preprocessing**  
  - Convert game frames into grayscale  
  - Resize frames for efficient processing  
- **Feature Extraction**  
  - Identify important pixels (enemies, ammo, health)  
  - Use convolutional neural networks (CNNs) for visual input  

✅ **Training & Testing** 🎯  
- **Reward-Based Training**  
  - Rewards for shooting enemies, dodging attacks, collecting health packs  
  - Negative rewards for getting hit, missing shots, or dying  
- **Model Optimization**  
  - Hyperparameter tuning for better learning rate, batch size, and exploration strategies  
  - Experimenting with different reward structures  

✅ **Hardware & Deployment** ⚙️  
- **Cloud Training** – Using **Google Colab ( GCP )** for faster model training  
- **Local Testing** – Running trained AI models on a **high-performance GPU**  

---

## 🔍 **Project Workflow**  

1️⃣ **Setup the ViZDoom Environment** 🕹️  
   - Install dependencies and configure game settings  

2️⃣ **Collect Training Data** 🎥  
   - Capture thousands of frames and preprocess them  

3️⃣ **Build the AI Model** 🧠  
   - Implement reinforcement learning algorithms (**DQN, PPO, A3C**)  

4️⃣ **Train the AI** 🎓  
   - Let the model play against the **DOOM boss** and learn from **trial & error**  

5️⃣ **Optimize & Fine-Tune** ⚙️  
   - Adjust parameters, improve decision-making, and fine-tune rewards  

6️⃣ **Test AI Performance** 🏆  
   - Measure how effectively the AI adapts and improves against the boss  

7️⃣ **Final Evaluation & Report** 📜  
   - Analyze results, document progress, and present findings  

---

🎯 **Goal:**  
Create an **autonomous AI agent** that can strategically **defeat the DOOM boss** by learning from **reinforcement training**, leveraging **computer vision**, and making **intelligent decisions** in real-time.  
