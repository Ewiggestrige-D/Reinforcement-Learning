# ML Lecture 23-1: Deep Reinforcement Learning
[ML Lecture 23-1: Deep Reinforcement Learning](https://www.youtube.com/watch?v=W8XF3ME8G2I)

## Scenario of Reinforcement Learning

### Agent
    - oberserve: state of 'enviroment'
    - action: change the 'enviroment'  -> reward or panelty
    - Target: agent learns to take 'ACTIONS' to 'Maximize' expected reward

### Learning to palt Go
- Supervised vs Reinforcement
  - supervised: rely on expert pre-knowledege --  human interface/label
  - reinforecement: learing from experience: try and reward -- non human interface/label

### example: playing vedio game
- widely studies:
  - gym: [https://gym.openai.com](https://gym.openai.com)
  - Universe: [https://openai.com/bug/universe](https://openai.com/bug/universe)
Machine learns to play video games as human players
 - what machine observes is pixels
 - machine learns to take  proper action itself

Difficulties of Reinforcement Learing
- reward delay
  - In space invader, only 'fire' obtains reward. although the moving before 'fire' is important
  - In Go playing, it may be better to sacrifice **immediate reward** to gain more **long-term** reward
  - Agent's actions affect the subsequent data it receives
    - exploration
  
## Outline
- Policy Base: learning an **actor**
- Value Base : learning a **critic**
- Combined: **Actor+Critic**:Asychronous Advantage Actor-Critic(A3C)
  - [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
### Polcy-Based Approch: Learning An Actor
Actor/Policy：$\text{Actor} = \pi (\text{Observation})$
其中Observation 是function input  
action是function output  
reward used to pick the **best function $\pi$**

**3 steps for deep learning**
1. define a set of funciton
2. goodness of function(loss)
3. pick the best function(minimize the loss)

#### **Neural Network** as Actor
- Input of neural network: the observation of machine represented as a vector or a matrix
- Output of neural network: each action corresponds to a neuron in output layer

#### **Goodne**ss of Actor
- Given an actor $\pi_{\theta(s)}$ with network parameter $\theta$
- Use the actor $\pi_{\theta(s)}$ to play the game
  - Total Reward:$R_{\theta} =\sum_{t=1}^{T}r_t$, 其中$r_t$是在timestep t获得的reward
  - Even with same actor,$R_{\theta}$ is different each time for 
    - randomness in the actor and the game
    - stochastic property
  - define $\overline{R_{\theta}}$ as the **expected value** of $R_{\theta}$
  - $\overline{R_{\theta}}$ evaluates the goodness of an actor $\pi_{\theta(s)}$ 
- An epsidode is considered as a trajectory/sequence $\tau$
  - $\tau={s_1,a_1,r_1,...,s_T,a_T,r_T}$, 其中$s_T$为时间T时候的observe的state，$a_T$为时间T时候采取的action，$r_T$是采取action 之后的reward
  - $R(\tau) = \sum_{t=1}^{T}r_t$
  - if you use an actor to play the game, each $\tau$ has a probability to be sampled:
    - probability depends on actor params $\theta$: $P(\tau|\theta)$
  - $\overline{R_{\theta}} = \sum_{\tau}R(\tau) P(\tau|\theta)$
    - Use $\pi_{\theta}$ to play game N times, obtain ${\tau^1,...,\tau^N}$ sampling $\tau$ from $P(\tau|\theta)$ N times
    - thus $\overline{R_{\theta}} = \sum_{\tau}R(\tau) P(\tau|\theta) \approx \frac{1}{N}\sum_{n=1}^{N}R(\tau^n)$

#### Pick the **best function**
- Gradient Descent
  - problem statement: $\theta^{*} = arg \max_{\theta} \overline{R_{\theta}} $
  - Gradient Descent: $\theta_1 \leftarrow \theta^0 + \eta \nabla \overline{R_{\theta^0}}$
  - 其中
  $$\nabla \overline{R_{\theta}} = 
  \begin{bmatrix} 
  \frac{\partial \overline{R_{\theta^0}}}{\partial w_1} \\ 
  \frac{\partial \overline{R_{\theta^0}}}{\partial w_2} \\ 
  \dots \\
  \frac{\partial \overline{R_{\theta^0}}}{\partial w_n} \\
  \frac{\partial \overline{R_{\theta^0}}}{\partial b_1} \\
  \frac{\partial \overline{R_{\theta^0}}}{\partial b_2} \\
  \dots \\
  \frac{\partial \overline{R_{\theta^0}}}{\partial b_n} \\
  \end{bmatrix}$$

  - 实际中
  $$\nabla \overline{R_{\theta}} = 
  \sum_{\tau}R(\tau) \nabla P(\tau|\theta)$$
$R(\tau)$ do not have to be differentiable
  - thus 
  $$\begin{aligned}
    \nabla \overline{R_{\theta}} & = 
  \sum_{\tau}R(\tau) \nabla P(\tau|\theta)\\
  & = \sum_{\tau}R(\tau)  P(\tau|\theta) \frac{\nabla P(\tau|\theta) }{P(\tau|\theta) } \\
  & = \sum_{\tau}R(\tau) P(\tau|\theta) \nabla_{\theta} \log P(\tau|\theta) \\
  & \approx \frac{1}{N}\sum_{n=1}^{N}R(\tau) \nabla_{\theta} \log P(\tau|\theta)
  \end{aligned}$$
  - 如何计算stochastic probability：
  $$P(\tau|\theta) =p(s_1) p(a_1|s_1,\theta)p(r_1,s_2|s_1,a_1)p(a_1|s_2,\theta)p(r_2,s_3|s_2,a_2)\dots \\
  =p(s_1) \prod_{t=1}^{T}p(a_t|s_t,\theta)p(r_t,s_{t+1}|s_t,a_t) $$

  $$logP(\tau|\theta) = \log p(s_1) +\sum_{t=1}^{T} \left( \log p(a_t|s_t,\theta) + \log p(r_t,s_{t+1}|s_t,a_t)\right) $$

  $$\nabla logP(\tau|\theta) = \sum_{t=1}^{T} \nabla \log p(a_t|s_t,\theta)$$
  
  - 化简之后
  $$\begin{aligned}
    \nabla \overline{R_{\theta}} 
  & \approx \frac{1}{N}\sum_{n=1}^{N}R(\tau) \nabla_{\theta} \log P(\tau|\theta)\\
  & =  \frac{1}{N}\sum_{n=1}^{N}R(\tau)  \sum_{t=1}^{T} \nabla \log p(a_t|s_t,\theta)\\
  & = \frac{1}{N}\sum_{n=1}^{N}  \sum_{t=1}^{T} R(\tau)\nabla \log p(a_t|s_t,\theta)
  \end{aligned}$$
  - 意义：if in $\tau^n$ machine takes $a_{t}^{n}$ when seeing $s_{t}^{n}$ in
    - $R(\tau^n)$ is *positive* -> Tuning $\theta$ to **increase** $p(a_t|s_t,\theta)$
    - $R(\tau^n)$ is *negative* -> Tuning $\theta$ to **decrease** $p(a_t|s_t,\theta)$
- **Add a baseline** 
  为了避免$R(\tau)$都是正数，而sampling次数有限导致的sample并非全部都是最优解,可以手动减掉一个baseline值（hyper-parameter），来保证策略会保持在一个最低优化的策略上
  $$\nabla \overline{R_{\theta}} 
  = \frac{1}{N}\sum_{n=1}^{N}  \sum_{t=1}^{T} (R(\tau)-b)\nabla \log p(a_t|s_t,\theta)
  $$


# ML Lecture 23-2: Policy Gradient (Supplementary Explanation)
[ML Lecture 23-2: Policy Gradient (Supplementary Explanation)](https://www.youtube.com/watch?v=y8UPGr36ccI&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=35)

主要分析了公式
  $$\begin{aligned}
    \nabla \overline{R_{\theta}} 
  & \approx \frac{1}{N}\sum_{n=1}^{N}R(\tau) \nabla_{\theta} \log P(\tau|\theta)\\
  & =  \frac{1}{N}\sum_{n=1}^{N}R(\tau)  \sum_{t=1}^{T} \nabla \log p(a_t|s_t,\theta)\\
  & = \frac{1}{N}\sum_{n=1}^{N}  \sum_{t=1}^{T} R(\tau)\nabla \log p(a_t|s_t,\theta)
  \end{aligned}$$
的意义 ，即将每一次的action看成一个classifier，**每一个classifier的先验概率来自于贝叶斯统计基于当前state做出的观察**，因此将时序问题分解为多分类问题。再类比多分类问题的**Loss Function: CrossEntropy**，将每一次action series的reward累加，并除以ssampling次数，作为累计reward的期望。


# ML Lecture 23-3: Reinforcement Learning (including Q-learning)
[ML Lecture 23-3: Reinforcement Learning (including Q-learning)](https://www.youtube.com/watch?v=2-JNBzCq77c&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=36)

## Learning to interact with Enviroment
- **How to slove this Problem**
    - Network as a function,learn as typical **supervised tasks**: pre-training ob expert knowledge -> "*behavior clone*" .
    <u>But machine do not know some behavior must copy, some can be ignored</u>

- Characteristics of Interaction
  - Agent's actions **affect the subsequent data it receives**
  - **Reward delay**

- 2 Learning Scenarios
  - Scenario 1: **Reinforcement Learning**
    - machine interacts with environment
    - machine obtains the reward from the **environment**, so it knows performance is **good or bad**
  - Scenario 2: **Learning by demonstration**
    - also known as '*imitation learning*',apprecnticeship learning
    - an expert demonstrates how to solve the task and machine learns from the demonstration

## Outline
- Reinforcement Learning
  - Training an actor
  - Training a Critic
  - Actor + Critic
- Inverse Reinforcement Learning


### Reinforcement Learning
Basic Components：
- Actor
- Enviroment
- Reward Function

#### **Neural Network** as Actor
- Input of neural network: the observation of machine represented as a vector or a matrix
- Output of neural network: each action corresponds to a neuron in output layer


#### Critic
- a critic does nor determine the action
- Given an actor $\pi$, it evaluates how good the actor is 
- State value function $V^{\pi}(s)$
  - when using actor $\pi$, the *cumulated* reward expects to be ontained after seeing observsation(state) $s$


How to estimate $V^{\pi}(s)$
- **Monte-Carlo** based approch (Sparse Reward)
  - the cirtic watches $\pi$ playing games:
  after seeing $s_a$:
    until the end of the episode, the cumulated reward is $G_a$

- **Temporal-Difference** Approach (Dense Reward)
  - at any time $t$: get
    - obervation $s_t$
    - action $a_t$
    - reward after this action $r_t$
    - new state $s_{t+1}$
    $V^{\pi}(s_{t})=V^{\pi}(s_{t+1})+r_t$

---


## 1. **目标：估计 $ V^{\pi}(s) $**

$ V^{\pi}(s) $ 是在策略 $ \pi $ 下从状态 $ s $ 开始的**期望回报（expected return）**：

$$
V^{\pi}(s) = \mathbb{E}_{\pi} \left[ G_t \mid S_t = s \right], \quad \text{其中 } G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
$$

---

## 2. **Monte Carlo (MC) 方法**

- **特点**：
  - 需要等到一个 episode 完全结束，才能计算完整的 return $ G_t $。
  - 用实际观测到的完整轨迹的累计折扣回报作为 $ V^{\pi}(s_t) $ 的目标值。
- **更新规则（例如 every-visit MC）**：
  $$
  V(s_t) \leftarrow V(s_t) + \alpha \left( G_t - V(s_t) \right)
  $$
- **适用场景**：
  - 特别适合 **稀疏奖励（sparse reward）** 环境，因为奖励只在 episode 结束时出现（如胜负结果），中间没有即时反馈。
  - 如果 reward 很稀疏，TD 方法可能难以学习（因为中间没有信号），而 MC 能直接利用最终结果。

✅ 所以：**MC 更适合 sparse reward** —— 这个说法基本正确。

---

## 3. **Temporal Difference (TD) 方法（如 TD(0)）**

- **特点**：
  - **不需要等待 episode 结束**，每一步都可以更新。
  - 使用 **bootstrapping**：用当前估计的 $ V(s_{t+1}) $ 来估计 $ V(s_t) $。
- **正确更新公式是**：
  $$
  V(s_t) \leftarrow V(s_t) + \alpha \left( r_t + \gamma V(s_{t+1}) - V(s_t) \right)
  $$
  其中 $ r_t + \gamma V(s_{t+1}) $ 是 **TD target**，而你写的
  > $ V^{\pi}(s_{t+1}) = V^{\pi}(s_t) + r_t $

  是**错误的**。这混淆了状态价值之间的关系。正确的贝尔曼方程是：
  $$
  V^{\pi}(s_t) = \mathbb{E}_{\pi} \left[ r_t + \gamma V^{\pi}(s_{t+1}) \right]
  $$

- **适用场景**：
  - 当环境中每一步都有有意义的 reward（即 **dense reward**），TD 可以快速学习，因为它每步都获得反馈。
  - 即使在 sparse reward 环境中，TD 也能工作（比如 DQN 在 Atari 游戏中处理稀疏得分），但可能收敛更慢或需要辅助技巧（如 reward shaping、n-step returns 等）。

✅ 所以：**TD 更适合 dense reward** —— 这个经验性说法有一定道理，但**不是绝对的**。TD 并不 *require* dense reward；只是在 dense reward 下表现更好、更稳定。

---

## 4. **Sparse vs Dense Reward ≠ MC vs TD**

更准确地说：

| 特性 | Monte Carlo | Temporal Difference |
|------|-------------|---------------------|
| 是否需要完整 episode | ✅ 是 | ❌ 否 |
| 是否使用 bootstrapping | ❌ 否 | ✅ 是 |
| 方差 | 高（因完整 return 随机性强） | 低（因用了估计值） |
| 偏差 | 无偏（如果策略固定） | 有偏（因 bootstrapping） |
| 对 sparse reward 的适应性 | 较好（直接用最终 reward） | 可能较差（需 long credit assignment） |

> 📌 **关键点**：sparse/dense reward 描述的是**环境特性**，而 MC/TD 是**学习算法的选择**。两者有关联，但不是一一对应。

---

## 5. 实践中的折中：**n-step returns / TD(λ)**

为了兼顾 MC 和 TD 的优点，常用：
- **n-step TD**：看未来 n 步的 reward 再更新（n=∞ 就是 MC，n=1 就是 TD(0)）
- ** eligibility traces (TD(λ))**：对所有 n-step return 加权平均

这些方法在 sparse reward 环境中往往比纯 TD(0) 更有效。

---

### ✅ 总结

- 你说“MC 对应 sparse reward，TD 对应 dense reward”是一种**经验上的简化说法**，有一定道理，但**不严格成立**。
- **MC 不依赖中间 reward，适合 sparse reward** ✔️  
- **TD 利用即时 reward，dense reward 下效率更高** ✔️  
- 但 TD 也能用于 sparse reward（只是 credit assignment 更难）
- 你写的 TD 更新公式是错的，正确形式是：  
  $$
  V(s_t) \leftarrow V(s_t) + \alpha \big( r_t + \gamma V(s_{t+1}) - V(s_t) \big)
  $$

---

#### Another Critic(Q-Function)
- state-action value fuction $Q^{\pi}(s,a)$
  - when using actor $\pi$, the cumulated reward expects to be obtained after seeing observation $s$ and taking action $a$
- **Q-Learning** 策略改进定理（Policy Improvement Theorem）
  - Given  $Q^{\pi}(s,a)$, find a new actor $\pi^\prime$ "**better**" than $\pi$
    - "Better": $V^{\pi^{\prime}}(s)>V^{\pi}(s)$ for **all state** $s$
    $$\pi^{\prime} =arg \max_{a}Q^{\pi}(s,a)$$
  - Flaws：|
    1. $\pi^\prime$ does not have extra parameters. It depends on Q
    2. not suitable for continous action $a$
  - Proof：
    - for $\pi^{\prime} =arg \max_{a}Q^{\pi}(s,a)$， $V^{\pi^{\prime}}(s)>V^{\pi}(s)$ for **all state** $s$
    - $V^{\pi}(s)=Q^{\pi}(s,\pi(s))\leq \max_{a}Q^{\pi}(s,a) =Q^{\pi}(s,\pi^{\prime}(s))$
    - thus
    $$\begin{aligned}
    V^{\pi^{\prime}}(s) & \leq Q^{\pi}(s,\pi^{\prime}(s)) \\
    & = \mathbb{E}_{\pi^{\prime}}[r_{t+1}+V^{\pi}(s_{t+1})|s_t=s] \\
    & \leq \mathbb{E}_{\pi^{\prime}}[r_{t+1}+Q^{\pi}(s_{t+1},\pi^{\prime}(s_{t+1}))|s_t=s] \\
    & = \mathbb{E}_{\pi^{\prime}}[r_{t+1}+r_{t+2}+V^{\pi}(s_{t+2})|s_t=s] \\
    & \leq \mathbb{E}_{\pi^{\prime}}[r_{t+1}+r_{t+2}+Q^{\pi}(s_{t+2},\pi^{\prime}(s_{t+2}))|s_t=s] \\
    & \dots \leq V^{\pi^{\prime}}(s)
    \end{aligned}$$


    你提供的这段内容是在**策略改进定理（Policy Improvement Theorem）**的框架下，试图证明：  
> 如果我们根据当前策略 $\pi$ 的动作价值函数 $Q^{\pi}(s, a)$ 构造一个新策略 $\pi'$，使得  
> $$
\pi'(s) = \arg\max_a Q^{\pi}(s, a),
$$  
> 那么新策略 $\pi'$ 一定 **不劣于**（甚至严格优于）原策略 $\pi$，即对所有状态 $s$，有 $V^{\pi'}(s) \geq V^{\pi}(s)$。

但你给出的“证明”存在一些**逻辑跳跃、符号混淆和不严谨之处**。下面我们一步步拆解、纠正并清晰地讲解这个经典结论。

---

## 🔷 背景：策略改进定理（Policy Improvement Theorem）

这是动态规划和强化学习中的一个核心结果，最早出现在 Sutton & Barto 的《Reinforcement Learning: An Introduction》中。

### 定义回顾：
- $V^{\pi}(s)$：在策略 $\pi$ 下从状态 $s$ 开始的期望回报。
- $Q^{\pi}(s, a) = \mathbb{E}_{\pi}[r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots \mid s_t = s, a_t = a]$
- 策略 $\pi$ 是一个概率分布：$\pi(a|s) = P(a_t = a \mid s_t = s)$

---

## 🔷 目标

给定一个策略 $\pi$ 和其对应的 $Q^{\pi}(s, a)$，定义一个**确定性贪心策略**：
$$
\pi'(s) = \arg\max_a Q^{\pi}(s, a)
$$
我们要证明：
$$
V^{\pi'}(s) \geq V^{\pi}(s), \quad \forall s
$$

如果对某些 $s$ 有严格不等式，则 $\pi'$ 严格优于 $\pi$。

---

## 🔷 正确的证明思路（简洁版）

### 第一步：比较 $V^{\pi'}(s)$ 和 $Q^{\pi}(s, \pi'(s))$

注意：$V^{\pi'}(s)$ 是在**新策略 $\pi'$** 下的值函数，而 $Q^{\pi}(s, a)$ 是在**旧策略 $\pi$** 下的动作价值。

但我们可以利用贝尔曼方程展开：

$$
V^{\pi'}(s) = \mathbb{E}_{a \sim \pi'(\cdot|s)} \left[ Q^{\pi'}(s, a) \right] = Q^{\pi'}(s, \pi'(s)) \quad (\text{因为 }\pi'\text{ 是确定性的})
$$

但这对我们没直接帮助，因为我们不知道 $Q^{\pi'}$。

### 关键技巧：用 $Q^{\pi}$ 来**下界估计** $V^{\pi'}$

考虑从状态 $s$ 出发，**第一步按 $\pi'$ 行动**，之后**继续按原策略 $\pi$ 行动**。这种混合策略的回报为：
$$
Q^{\pi}(s, \pi'(s)) = \mathbb{E}_{\pi} \left[ r_t + \gamma V^{\pi}(s_{t+1}) \mid s_t = s, a_t = \pi'(s) \right]
$$

由于 $\pi'(s) = \arg\max_a Q^{\pi}(s, a)$，所以：
$$
Q^{\pi}(s, \pi'(s)) \geq Q^{\pi}(s, \pi(s)) = V^{\pi}(s)
\tag{1}
$$

但这只是“走一步 $\pi'$，后面跟 $\pi$”的回报，不是完整的 $V^{\pi'}(s)$。

### 第二步：递归展开（核心思想）

现在考虑完整执行 $\pi'$ 的回报 $V^{\pi'}(s)$。我们可以写成：

$$
V^{\pi'}(s) = \mathbb{E}_{\pi'} \left[ r_t + \gamma V^{\pi'}(s_{t+1}) \mid s_t = s \right]
$$

但我们想把它和 $Q^{\pi}$ 联系起来。为此，使用**归纳法或逐次展开**：

从定义出发：
$$
V^{\pi'}(s) = \mathbb{E}_{\pi'} \left[ r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots \mid s_t = s \right]
$$

而另一方面，考虑下面这个不等式链（这才是你原文试图表达的，但写错了）：

---

## 🔷 正确的不等式链（修正你的推导）

我们从 $V^{\pi'}(s)$ 出发，但用 $Q^{\pi}$ 逐步展开：

由于 $\pi'(s) = \arg\max_a Q^{\pi}(s, a)$，我们有：
$$
V^{\pi'}(s) = Q^{\pi'}(s, \pi'(s)) \geq Q^{\pi}(s, \pi'(s)) \quad ??
$$

❌ 这不对！$Q^{\pi'}$ 和 $Q^{\pi}$ 是不同策略下的函数，不能直接比较。

✅ 正确做法是：**证明 $V^{\pi'}(s) \geq V^{\pi}(s)$ 通过递归不等式**。

---

### ✅ 标准证明（来自 Sutton & Barto）

我们证明：对任意 $s$，
$$
V^{\pi}(s) \leq Q^{\pi}(s, \pi'(s)) \leq V^{\pi'}(s)
$$

**第一步**：由 $\pi'$ 的定义，
$$
Q^{\pi}(s, \pi'(s)) = \max_a Q^{\pi}(s, a) \geq Q^{\pi}(s, \pi(s)) = V^{\pi}(s)
\tag{A}
$$

**第二步**：现在看 $V^{\pi'}(s)$。将其展开一步：
$$
V^{\pi'}(s) = \mathbb{E}_{\pi'} \left[ r_t + \gamma V^{\pi'}(s_{t+1}) \mid s_t = s \right]
= r(s, \pi'(s)) + \gamma \sum_{s'} P(s'|s, \pi'(s)) V^{\pi'}(s')
$$

同样，把 $Q^{\pi}(s, \pi'(s))$ 展开：
$$
Q^{\pi}(s, \pi'(s)) = r(s, \pi'(s)) + \gamma \sum_{s'} P(s'|s, \pi'(s)) V^{\pi}(s')
\tag{B}
$$

现在假设**归纳假设**：对所有 $s'$，有 $V^{\pi'}(s') \geq V^{\pi}(s')$。

那么从 (B) 可得：
$$
Q^{\pi}(s, \pi'(s)) \leq r(s, \pi'(s)) + \gamma \sum_{s'} P(s'|s, \pi'(s)) V^{\pi'}(s') = V^{\pi'}(s)
\tag{C}
$$

结合 (A) 和 (C)：
$$
V^{\pi}(s) \leq Q^{\pi}(s, \pi'(s)) \leq V^{\pi'}(s)
$$

因此 $V^{\pi'}(s) \geq V^{\pi}(s)$ 对所有 $s$ 成立。

**基础情况**：当 horizon 为 0（无未来 reward），显然成立。  
**归纳完成** → 策略改进定理得证。

---

## 🔷 你原文中的问题分析

你写的推导：

> $V^{\pi^{\prime}}(s)=Q^{\pi}(s,\pi(s))\leq \max_{a}Q^{\pi}(s,a) =Q^{\pi}(s,\pi^{\prime}(s))$

这里第一个等号就错了：
- 左边是 $V^{\pi'}(s)$（新策略的值）
- 右边是 $Q^{\pi}(s, \pi(s)) = V^{\pi}(s)$（旧策略的值）

所以应该是：
$$
V^{\pi}(s) = Q^{\pi}(s, \pi(s)) \leq \max_a Q^{\pi}(s, a) = Q^{\pi}(s, \pi'(s))
$$

但你写成了 $V^{\pi'}(s) = Q^{\pi}(s, \pi(s))$，这是混淆了策略。

后面的展开：
> $Q^{\pi}(s,\pi^{\prime}(s)) = \mathbb{E}_{\pi^{\prime}}[r_{t+1}+V^{\pi}(s_{t+1})|s_t=s]$

这也不准确。正确的应是：
$$
Q^{\pi}(s, a) = \mathbb{E}_{\text{environment}} [ r_t + \gamma V^{\pi}(s_{t+1}) \mid s_t = s, a_t = a ]
$$
这里的期望**只关于环境转移**，**不关于策略 $\pi'$**！因为 $Q^{\pi}$ 是固定策略 $\pi$ 下的函数，采取动作 $a$ 后，后续仍按 $\pi$ 执行。

所以不应写 $\mathbb{E}_{\pi'}$，而应写 $\mathbb{E}$ 或 $\mathbb{E}_{P}$。

最后你得到一个循环不等式 “$\dots \leq V^{\pi'}(s)$”，逻辑上不成立。

---

## 🔷 总结

| 内容 | 正确理解 |
|------|--------|
| **策略改进** | $\pi'(s) = \arg\max_a Q^{\pi}(s, a)$ ⇒ $V^{\pi'}(s) \geq V^{\pi}(s)$ |
| **关键不等式** | $V^{\pi}(s) \leq Q^{\pi}(s, \pi'(s)) \leq V^{\pi'}(s)$ |
| **证明方法** | 利用贝尔曼方程 + 数学归纳法（或逐次展开） |
| **你原文错误** | 混淆了 $V^{\pi'}$ 和 $Q^{\pi}$，误用了期望符号，逻辑循环 |
| **适用性** | 该定理适用于**任何 MDP**，无论 reward sparse/dense |

---

## 🔷 附加说明：Q-Learning 与策略改进

- Q-learning 是一种 **off-policy** 方法，它学习的是**最优 Q 函数 $Q^*$**，而不是某个 $\pi$ 的 $Q^{\pi}$。
- 当 Q-learning 收敛后，$\pi^*(s) = \arg\max_a Q^*(s, a)$ 就是最优策略。
- 但在**策略迭代（Policy Iteration）**中，我们交替进行：
  1. **策略评估（Policy Evaluation）**：计算 $Q^{\pi}$
  2. **策略改进（Policy Improvement）**：令 $\pi' = \arg\max_a Q^{\pi}(s, a)$

你描述的内容更接近**策略迭代中的策略改进步骤**，而不是 Q-learning 本身。

