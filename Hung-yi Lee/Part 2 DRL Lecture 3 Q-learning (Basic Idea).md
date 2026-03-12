# Part 2 DRL Lecture 3 Q-learning (Basic Idea)
[Lecture 3 Q-learning (Basic Idea)](https://www.youtube.com/watch?v=o_g9JUMw1Oc)

## Q-Learning

### Introduction of **Q-Learning**

**Critic**
- A critic does not directly determine the action.
- Given an actor $\pi$, it evaluates how good the actor is
- State value function $V^{\pi}(s)$
  - When using actor $\pi$, the *cumulated* reward expects to be obtained after visiting state $s$


> Note: Critic is a evaluating funciton, so Critic needs an actor to perform in the scenario to get rewards

**How to estimate $V^{\pi}(s)$**
- **Monte-Carlo(MC)** based Approach
  - The critic watches actor $\pi$ playing the game
  - After seeing $s_a$/$s_b$,Until the **end of the episode**, the cumulated reward is $G_a$/$G_b$
- **Temporal difference(TD)** Based approach
  - for any time $t$, actor observe the state $s_t$,and take an action $a_t$, get a reward $r_t$ and new state $s_{t+1}$
  $V^{\pi}(s) = V^{\pi}(s+1)+r_t$
  - why **Temporal difference(TD)** Based approach: Some applications have very long episodes, so that **delaying all learning until an episode's end** is too *slow*.
- MC v.s . TD
![MCvsTD](/Hung-yi%20Lee/img/Part%202%20DRL%20Lecture%203/MCvsTD.png)

**Anoher Critic State-Action Value Function**
- State action value function $Q^{\pi}(s,a)$
  - When using actor 𝜋, the *cumulated* reward expects to be obtained after taking $a$ at state $s$
  - 需要注意的是actor 𝜋在看见state $s$的时候不一定要采取action $a$，而$Q^{\pi}(s,a)$ 则评估的是，看见state $s$的时候强制采取action $a$所得到的accumulated reward。 接下来则让actor 𝜋按照policy自由采取动作。

**Another Way to use Critic: Q-Learning**
- 𝜋 interacts with the environment
- Learning $Q^{\pi}(s,a)$ (TD or MC)
- Find a new actor 𝜋′ “better” than 𝜋, and update

**Q-Learning**
- Given $Q^{\pi}(s,a)$, find a new actor 𝜋′ “better” than 𝜋
  - **Better**: $V^{\pi'}(s)> V^{\pi}(s)  $, for all state $s$
  $$\pi'(s) = arg \max_{a}Q^{\pi}(s,a)$$
    - 𝜋′does not have extra parameters. It depends on Q
    - Not suitable for continuous action $a$

**公式推导**
对任意 $s$，
$$
V^{\pi}(s) \leq Q^{\pi}(s, \pi'(s)) \leq V^{\pi'}(s)
$$
> 利用贪心算法的思想，将每一步的action都取最优 $Q^{\pi}(s, \pi'(s)) = \max_a Q^{\pi}(s, a)$


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

### Tips of **Q-Learning**
1. **Target Network**
  - For Temporal differential based approach, in serial $\dots {s_t,a_t,r_t,s_{t+1}} \dots$ we have $Q^{\pi}(s,a) = r_t+ Q^{\pi}(s_{t+1},\pi(s_{t+1}))$
  - to update $Q^{\pi}(s,a)$ and get the fixed **Target function** $r_t+ Q^{\pi}(s_{t+1},\pi(s_{t+1}))$, but the parameters of $Q^{\pi}$ is share, i.e. when $Q^{\pi}(s,a)$ is updated, $r_t+ Q^{\pi}(s_{t+1},\pi(s_{t+1}))$ is also changed
  - **Solution**: use the **fixed** $r_t+ Q^{\pi}(s_{t+1},\pi(s_{t+1}))$ as target and update $Q^{\pi}(s,a)$ params several times meawhile update $r_t+ Q^{\pi}(s_{t+1},\pi(s_{t+1}))$ params
2. **Exploration**
   -  Policy based on Q-function $a = arg \max_{a}Q(s,a)$. This is not a good way for data collection. *The actions become fixed if one of the action get huge reward in this state*
   -  **Solution**:
      -  **Epsilon Greedy**:
      $$ a = \left\{ \begin{aligned} 
      & arg \max_{a} Q(s,a),\quad \text{with probability} 1-\varepsilon, \\
          & ramdom, \quad otherwise\\
      \end{aligned}
      \right.
      $$
      where $\varepsilon$ would decay during learning.即，初始的时候，actor采取各种action的概率相等（平均初始化），开始的时候actor会有更高的概率选择reward高的action，训练多个epoch之后，actor选择其他reward的几率$\varepsilon$会被逐渐降低
      -  **Boltzmann Exploration**:
      $P(a|s) = \frac{\exp(Q(s,a))}{\sum_{a}\exp(Q(s,a))}$
3. **Replay Buffer**
   - Put the experience into **buffer** .
   - The experience in the buffer comes from **different policies**(off-policy). Drop the old experience if the buffer is full.
   - Use multiple experience to learn the better policy $\pi$
   - experience from different as **Regularization terms** to gain more **robustness**
4. **Typical Q-Learning Algorithm**
   1. Initialize Q-function $Q$, target Q-function $\hat{Q} =Q$ at initial
   2. In each epoch, for each time step $t$\
      1. Given state $s_t$, take action $a_t$ based on $Q$ (epsilon greedy/Boltzmann Exploration)
      2. Obtain reward $r_t$, and reach new state $s_{t+1}$
      3. Store $(s_t,a_t,r_t,s_{t+1})$ into buffer
      4. Sample $(s_i,a_i,r_i,s_{i+1})$ from buffer (usually a batch)
      5. Target $y = r_i+\max_{a}\hat{Q}(s_{i+1},a)$
      6. Update the parameters of $Q$ to make $Q(s_i,a_i)$,close to Target $y$ (regression)
      7. Every $N$ steps reset $\hat{Q} \leftarrow Q$ ($N$ is a **hyperparameter**)

### **Q-Learning** for continious actions


## Appendix  状态价值函数与状态-动作价值函数的差异

- **状态价值函数**：$ V^{\pi}(s) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^\infty \gamma^t r_t \,\big|\, s_0 = s \right] $
- **状态-动作价值函数**：$ Q^{\pi}(s,a) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^\infty \gamma^t r_t \,\big|\, s_0 = s, a_0 = a \right] $

确实都依赖于同一个策略 $ \pi $（即“绑定一个 actor”），但它们在**信息粒度、用途、训练目标和算法设计**上存在本质区别。

---

### 一、这两个 Critic 的差别是什么？

| 维度 | $ V^{\pi}(s) $ | $ Q^{\pi}(s,a) $ |
|------|------------------|--------------------|
| **输入** | 仅状态 $ s $ | 状态 $ s $ + 动作 $ a $ |
| **含义** | “从状态 $ s $ 开始，按策略 $ \pi $ 走，未来总回报期望” | “在状态 $ s $ 下执行动作 $ a $，之后按策略 $ \pi $ 走，未来总回报期望” |
| **是否显式依赖动作** | 否 | 是 |
| **是否可用于直接选动作** | 否（需知道 $ \pi(a\|s) $ 才能评估动作） | 是（可比较不同 $ a $ 的 $ Q $ 值） |
| **与策略的关系** | 隐式包含策略（通过后续动作分布） | 显式包含初始动作，后续仍按 $ \pi $ |

> ✅ **关键区别**：  
> $ Q^{\pi}(s,a) $ **多了一个自由度**——它告诉你“如果我现在偏离策略 $ \pi $，强行执行某个动作 $ a $，会怎样”。而 $ V^{\pi}(s) $ 只描述“按当前策略走”的结果。

---

### 二、这种差异体现在什么地方？（在计算或公式构建时）

#### 1. **Bellman 方程不同**

- $ V^{\pi}(s) $ 的 Bellman 方程：
  $$
  V^{\pi}(s) = \mathbb{E}_{a \sim \pi(\cdot|s)} \left[ r(s,a) + \gamma \mathbb{E}_{s' \sim P(\cdot|s,a)} [V^{\pi}(s')] \right]
  $$

- $ Q^{\pi}(s,a) $ 的 Bellman 方程：
  $$
  Q^{\pi}(s,a) = r(s,a) + \gamma \mathbb{E}_{s' \sim P(\cdot|s,a)} \left[ \mathbb{E}_{a' \sim \pi(\cdot|s')} [Q^{\pi}(s',a')] \right]
  $$

👉 注意：$ Q $ 的更新**不需要对当前动作求期望**，因为 $ a $ 已给定；而 $ V $ 必须对 $ a \sim \pi $ 求期望。

#### 2. **优势函数（Advantage Function）的构造**

优势函数定义为：
$$
A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)
$$

- 它衡量“执行动作 $ a $ 相比于按策略 $ \pi $ 平均执行，好多少”。
- 在 **A2C、A3C、PPO** 等算法中，**必须同时估计 $ V $ 和 $ Q $（或直接估计 $ A $）**。
- 如果只用 $ V $，无法直接得到 $ A $；如果只用 $ Q $，可以推导出 $ V $：  
  $$
  V^{\pi}(s) = \mathbb{E}_{a \sim \pi(\cdot|s)} [Q^{\pi}(s,a)]
  $$

#### 3. **目标网络与损失函数设计**

- 使用 $ V $ 的 Critic（如 A2C）：损失为 $ (R - V(s))^2 $
- 使用 $ Q $ 的 Critic（如 DQN、DDPG）：损失为 $ (R - Q(s,a))^2 $

⚠️ **注意**：在连续动作空间中，$ Q(s,a) $ 是关于 $ a $ 的函数，通常需要额外处理（如 DDPG 中用确定性策略）。

---

### 三、这个差别会对 DRL 造成什么样的影响？

| 影响维度 | 使用 $ V $ | 使用 $ Q $ |
|--------|------------|------------|
| **适用算法** | Actor-Critic 类（A2C, PPO, TRPO） | Value-based（DQN）或 Deterministic AC（DDPG, TD3） |
| **动作空间适应性** | 离散/连续均可（因不直接输出动作值） | 离散：DQN；连续：需配合策略网络（如 DDPG） |
| **采样效率** | 高（on-policy，但可结合 GAE） | Off-policy 方法（如 DQN）样本效率高 |
| **策略梯度计算** | 需要 $ A(s,a) $，通常用 $ R - V(s) $ 估计 | 在 DDPG 中，梯度通过 $ \nabla_a Q(s,a) $ 传给 Actor |
| **方差** | $ V $-based 优势估计方差较低（因减去了基线） | $ Q $-learning 可能有高估偏差（尤其 DQN） |

> 💡 **核心影响**：  
> - 如果你用 **随机策略（stochastic policy）**（如 Gaussian policy），通常搭配 **$ V $-critic**（因为 $ Q $ 在连续空间难优化）。  
> - 如果你用 **确定性策略（deterministic policy）**，则可用 **$ Q $-critic**（如 DDPG）。  
> - **PPO/A2C 等主流 on-policy 算法几乎都用 $ V $-critic + 优势函数**，而非直接学 $ Q $。

---

### 四、使用这两种 Critic 时应该注意什么问题？

#### ✅ 使用 $ V^{\pi}(s) $ 时注意：
1. **不能直接用于动作选择**：必须配合 Actor（策略网络）。
2. **优势估计的偏差**：若 $ V $ 估计不准，$ A = R - V(s) $ 会有偏差，影响策略更新。
3. **需确保 critic 与 actor 同步**：两者都基于同一策略 $ \pi $，若 critic 滞后，会导致错误的梯度信号。
4. **通常用 GAE（Generalized Advantage Estimation）** 来平衡 bias-variance。

#### ✅ 使用 $ Q^{\pi}(s,a) $ 时注意：
1. **维度灾难（离散动作）**：动作空间大时，$ Q $ 表格或网络输出维度过高。
2. **连续动作的优化困难**：需对 $ a $ 求梯度（如 DDPG 中 $ \nabla_\theta J \propto \mathbb{E}[\nabla_a Q \cdot \nabla_\theta \pi] $），要求 $ Q $ 对 $ a $ 光滑。
3. **过估计问题**：DQN 中的 max 操作导致 $ Q $ 高估，需用 Double DQN 或 Clipping（TD3）缓解。
4. **off-policy 稳定性**：需经验回放、目标网络等技巧防止发散。

---

### 五、补充：能否只用一个？能不能互相替代？

- **理论上可以转换**：
  - $ V^{\pi}(s) = \mathbb{E}_{a \sim \pi}[Q^{\pi}(s,a)] $
  - $ Q^{\pi}(s,a) = r(s,a) + \gamma \mathbb{E}_{s'}[V^{\pi}(s')] $（仅当后续严格按 $ \pi $ 执行）

- **实践中不建议随意替换**：
  - 在 **on-policy 算法**中，用 $ V $ 更稳定、高效。
  - 在 **off-policy + 离散动作**中，$ Q $ 更自然（如 DQN）。
  - 在 **连续控制**中，现代方法（SAC、TD3）通常同时维护 $ Q $ 和（隐式的）$ V $ 或使用 entropy-regularized $ V $。

---

### 总结

| 问题 | 回答要点 |
|------|--------|
| **差别是什么** | $ V $ 只看状态，$ Q $ 看状态+动作；$ Q $ 包含“反事实”动作评估能力 |
| **体现在哪** | Bellman 方程、优势函数构造、损失函数形式不同 |
| **对 DRL 的影响** | 决定算法类型（on/off-policy）、动作空间适配性、方差与偏差特性 |
| **使用注意事项** | $ V $：需配合 actor，注意优势估计；$ Q $：注意过估计、连续动作梯度、维度问题 |

> 📌 **一句话总结**：  
> **$ V $ 是“策略的自我评价”，$ Q $ 是“对所有可能动作的外部评价”**。  

