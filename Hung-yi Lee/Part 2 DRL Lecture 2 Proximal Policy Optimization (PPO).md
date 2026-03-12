# Part 2 DRL Lecture 2 Proximal Policy Optimization (PPO)
[Lecture 2 Proximal Policy Optimization (PPO)](https://www.youtube.com/watch?v=OAKAZhFmYoI)

## From **on-policy** to **off-policy**

### **On-policy v.s  Off-policy**
- **On-policy**: The agent learned and the agent interacting with the environment is the same. 训练与推理的agent是同一个agent（或者actor）
- **Off-policy**: The agent learned and the agent interacting with the environment is different. 训练与推理的agent不是同一个，即一个agent可以从现有的其他agent的经验中进行学习.或者单次训练数据可以被多次复用。


### **From On-policy to Off-policy**

#### **On-Policy**
  > $$\nabla \overline{R_{\theta}} = E_{\tau \sim p_{\theta}(\tau)}[R(\tau)\nabla \log p_{\theta}(\tau)]$$

- use $\pi_{\theta}$ to collect data. when $\theta$ is updated, we have to sample training data again.
- **Goal**: useing the sample from  $\pi_{\theta'}$ to train $\theta$. $\theta'$ is fixed, so we can re-use the sample data


### **Importance Sampling(sample from different distribution)**
when $x^{i}$ is sampleed from distribution $p(x)$
$$E_{x \sim p(x)}[f(x)] \approx \frac{1}{N}\sum_{i=1}^{N}f(x^{i})$$

if we use $x^{i}$ is sampleed from distribution $q(x)$
$$\begin{aligned}
    E_{x \sim p(x)}[f(x)] & \approx \frac{1}{N}\sum_{i=1}^{N}f(x^{i}) \\
    \rightarrow^{\text{continuous}} & = \int f(x)p(x)dx \\
    & = \int f(x)\frac{p(x)}{q(x)}q(x)dx \\
    & = E_{x \sim q(x)}[f(x)\frac{p(x)}{q(x)}]
\end{aligned}
$$
where $\frac{p(x)}{q(x)}$ is the importace weight that measure the appropriate weight from target distribution $p(x)$ mapping into the implemented distribution $q(x)$

**Issue of IMportance Sampling**
Even the expexction equals $E_{x \sim p(x)}[f(x)] =  E_{x \sim q(x)}[f(x)\frac{p(x)}{q(x)}]$,
the variance differs 
$$\begin{aligned}
    Var_{x \sim p}[f(x)] &= E_{x \sim p}[f^2(x)] - (E_{x \sim p}[f(x)])^2 \\
    Var_{x \sim q(x)}\left[f(x)\frac{p(x)}{q(x)}\right] & = E_{x \sim q(x)}\left[ \left(f(x)\frac{p(x)}{q(x)} \right)^2 \right] - \left( E_{x \sim q(x)}\left[f(x)\frac{p(x)}{q(x)}\right] \right)^2 \\
    & = E_{x \sim p(x)}\left[f^2(x)\frac{p(x)}{q(x)}  \right] - (E_{x \sim p(x)}[f(x)])^2 \\
\end{aligned}
$$ 

**详细推导**：

- 简化第一项（二次项）

考虑第一项：

$$
\mathbb{E}_{x \sim q(x)}\left[\left(f(x)\frac{p(x)}{q(x)}\right)^2\right] = \int \left(f(x)\frac{p(x)}{q(x)}\right)^2 q(x) dx
= \int f^2(x) \frac{p^2(x)}{q^2(x)} q(x) dx
= \int f^2(x) \frac{p^2(x)}{q(x)} dx
$$

现在我们希望将其写成关于 $ p(x) $ 的期望形式。注意到：

$$
\int f^2(x) \frac{p^2(x)}{q(x)} dx = \int f^2(x) \frac{p(x)}{q(x)} \cdot p(x) dx = \mathbb{E}_{x \sim p(x)}\left[ f^2(x) \frac{p(x)}{q(x)} \right]
$$

✅ **关键步骤**：我们将积分中的测度从 $ q(x)dx $ 转换为 $ p(x)dx $，通过提取一个 $ p(x) $ 出来作为新的概率测度，剩下的 $ \frac{p(x)}{q(x)} $ 作为权重。

因此，

$$
\mathbb{E}_{x \sim q(x)}\left[\left(f(x)\frac{p(x)}{q(x)}\right)^2\right] = \mathbb{E}_{x \sim p(x)}\left[ f^2(x) \frac{p(x)}{q(x)} \right]
$$

---

- 简化第二项（一次项）

考虑第二项：

$$
\mathbb{E}_{x \sim q(x)}\left[f(x)\frac{p(x)}{q(x)}\right] = \int f(x) \frac{p(x)}{q(x)} q(x) dx = \int f(x) p(x) dx = \mathbb{E}_{x \sim p(x)}[f(x)]
$$

这正是重要性采样的基本恒等式（取 $ g(x) = f(x) $）。

所以，

$$
\left( \mathbb{E}_{x \sim q(x)}\left[f(x)\frac{p(x)}{q(x)}\right] \right)^2 = \left( \mathbb{E}_{x \sim p(x)}[f(x)] \right)^2
$$


### **Difference of Variance**
对于 Importance Sampling，使用不同的distribution进行采样，
$$
\begin{aligned}
    \Delta Var & =  Var_{x \sim p}[f(x)] - Var_{x \sim q(x)}\left[f(x)\frac{p(x)}{q(x)}\right] \\
    & =  E_{x \sim p}[f^2(x)] - (E_{x \sim p}[f(x)])^2 - \left( E_{x \sim p(x)}\left[f^2(x)\frac{p(x)}{q(x)}  \right] - (E_{x \sim p(x)}[f(x)])^2  \right) \\
    & =  E_{x \sim p}[f^2(x)] -E_{x \sim p(x)}\left[f^2(x)\frac{p(x)}{q(x)}  \right] \\
    & = E_{x \sim p}\left[f^2(x)\left( 1- \frac{p(x)}{q(x)}\right)  \right] 
\end{aligned}
$$ 

即只有当$p(x)$与$q(x)$相差不大时，re-sampling的variance才会比较小，数据比较稳定。


![Importance Sampling from different Distribution](/Hung-yi%20Lee/img/Part%202%20DRL%20Lecture%202/Importance%20Sampling%20from%20different%20Distribution.png)



#### **On-Policy -> Off-Policy**
  > On-Policy: $$\nabla \overline{R_{\theta}} = E_{\tau \sim p_{\theta}(\tau)}[R(\tau)\nabla \log p_{\theta}(\tau)]$$

- use $\pi_{\theta}$ to collect data. when $\theta$ is updated, we have to sample training data again.
- **Goal**: useing the sample from  $\pi_{\theta'}$ to train $\theta$. $\theta'$ is fixed, so we can re-use the sample data

> Off-Policy : $$\nabla \overline{R_{\theta}} = E_{\tau \sim p_{\theta'}(\tau)}[\frac{p_{\theta}(\tau)}{p_{\theta'}(\tau)}R(\tau)\nabla \log p_{\theta}(\tau)]$$

- Sample from data $\theta'$
- Use policy $\pi(\theta')$ to train $\theta$ many times

**Gradient for Off-Policy update**
key formula: $\nabla f(x) = f(x)\nabla \log f(x)$

$$ \nabla \overline{R_{\theta,a_t}} =  \frac{1}{N}\sum_{n=1}^{N}  \sum_{t=1}^{T} A^{\theta}(s_t,a_t)\nabla \log p(a_t|s_t,\theta) $$

where **Advantage Function** defines as
$$A^{\theta}(s_t,a_t) = \sum_{t'=t}^{T}\gamma^{t'-t}\cdot r_{t'}^{n}-b$$
 where $\theta$ labels the ***chosen model/strategy that interacts with the environment***

Then Gradient for Off-Policy update should be
$$\begin{aligned}
    \nabla \overline{R_{\theta,a_t}} & =\nabla E_{(s_t,a_t)\sim \pi_{\theta}}\left[A^{\theta}(s_t,a_t)\nabla \log p(a_t|s_t,\theta) \right] \\
    \rightarrow^{\text{off-policy}} & = E_{(s_t,a_t)\sim \pi_{\theta'}}\left[\frac{P_{\theta}(s_t,a_t)}{P_{\theta'}(s_t,a_t)}A^{\theta'}(s_t,a_t)\nabla \log p(a_t|s_t,\theta) \right] \\
    & \text{assume} A^{\theta}(s_t,a_t) \approx A^{\theta'}(s_t,a_t) \\
    & = E_{(s_t,a_t)\sim \pi_{\theta'}}\left[\frac{P_{\theta}(s_t,a_t)}{P_{\theta'}(s_t,a_t)}A^{\theta}(s_t,a_t)\nabla \log p(a_t|s_t,\theta) \right] \\
    & = E_{(s_t,a_t)\sim \pi_{\theta'}}\left[\frac{P_{\theta}(a_t|s_t)}{P_{\theta'}(a_t|s_t)}\frac{P_{\theta}(s_t)}{P_{\theta'}(s_t)}A^{\theta}(s_t,a_t)\nabla \log p(a_t|s_t,\theta) \right] \\
    & \text{assume} \frac{P_{\theta}(s_t)}{P_{\theta'}(s_t)}\approx 1 \\
    \nabla J^{\theta'}(\theta)& = E_{(s_t,a_t)\sim \pi_{\theta'}}\left[\frac{P_{\theta}(a_t|s_t)}{P_{\theta'}(a_t|s_t)}A^{\theta}(s_t,a_t)\nabla \log p(a_t|s_t,\theta) \right] \\
    \leftarrow J^{\theta'}(\theta)& = E_{(s_t,a_t)\sim \pi_{\theta'}}\left[\frac{P_{\theta}(a_t|s_t)}{P_{\theta'}(a_t|s_t)}A^{\theta}(s_t,a_t) \right]
\end{aligned}
$$
其中和$\theta$有关的项仅有$P_{\theta}(a_t|s_t)$, 因此
$$\begin{aligned}
    \nabla J^{\theta'}(\theta)& =E_{(s_t,a_t)\sim \pi_{\theta'}}\left[\frac{\nabla P_{\theta}(a_t|s_t)}{P_{\theta'}(a_t|s_t)}A^{\theta}(s_t,a_t) \right]\\
    & = E_{(s_t,a_t)\sim \pi_{\theta'}}\left[\frac{P_{\theta}(a_t|s_t)}{P_{\theta'}(a_t|s_t)}A^{\theta}(s_t,a_t)\nabla \log p(a_t|s_t,\theta) \right] \\
\end{aligned}
$$

### Add a contriant on $\frac{P_{\theta}(a_t|s_t)}{P_{\theta'}(a_t|s_t)}$

- $\theta$ cannot be very different from $\theta'$
- contriant on **behavior** not parameters


**Proximal Policy Optimization (PPO)**
> $$J_{\text{PPO}}^{\theta'}(\theta) =J^{\theta'}(\theta) - \beta KL(\theta,\theta') $$
> where $KL(\theta,\theta')$ is the Kullback–Leibler Divergence and $\beta$ is learning rate


**Trust Region Policy Optimization (TRPO)**
> $$ \nabla J^{\theta'}(\theta) =E_{(s_t,a_t)\sim \pi_{\theta'}}\left[\frac{\nabla P_{\theta}(a_t|s_t)}{P_{\theta'}(a_t|s_t)}A^{\theta}(s_t,a_t) \right] $$
> where $KL(\theta,\theta') < \delta $ as a constraint


**Why KL Divergence on behavior**
   - 欧氏距离 $ \|\theta - \theta'\|^2 $ 忽略了策略函数的实际行为变化。(On Parameter)
   - 例如，两个参数相差很大，但策略输出几乎相同（如 softmax 后概率不变），此时 KL ≈ 0，而欧氏距离可能很大。(On behavior)
   - KL 更能反映**策略行为的实际差异**。

### PPO Algorithm
- Initial policy parameters $\theta^0$
- in each epoch
  - Using $\theta^k$to interact with the environment to collect ${s_t,a_t}$ and compute advantage $A^{\theta^{k}}(s_t,a_t)$
  - find $\theta$ optimizing $J_{\text{PPO}}(\theta)$
  $$J_{\text{PPO}}^{\theta^{k}}(\theta) = J^{\theta^{k}}(\theta) - KL(\theta,\theta^k)$$
  and update parameters several times
- **Adaptive KL penalty**
  - if $KL(\theta,\theta^k) > KL_{\max}$, increase $\beta$
  - if $KL(\theta,\theta^k) <> KL_{\min}$, decrease $\beta$

**PPO algorithm**
 $$J_{\text{PPO}}^{\theta^{k}}(\theta) = J^{\theta^{k}}(\theta) - KL(\theta,\theta^k)$$

where $J^{\theta^{k}}(\theta) \approx \sum_{(s_t,a_t)} \frac{ P_{\theta}(a_t|s_t)}{P_{\theta'}(a_t|s_t)}A^{\theta}(s_t,a_t)$


**PPO2 algorithm**
$$J_{\text{PPO2}}^{\theta^{k}}(\theta) \approx \sum_{(s_t,a_t)} \min \left( \frac{ P_{\theta}(a_t|s_t)}{P_{\theta'}(a_t|s_t)}A^{\theta}(s_t,a_t), clip \left( \frac{ P_{\theta}(a_t|s_t)}{P_{\theta'}(a_t|s_t)},1-\varepsilon,1+\varepsilon \right)A^{\theta}(s_t,a_t)\right)$$

Clip-Funtion
![clip-Function.png](/Hung-yi%20Lee/img/Part%202%20DRL%20Lecture%202/clip-Function.png)


PPO2 Function (Selective RuLU)
![PPO2 Function](/Hung-yi%20Lee/img/Part%202%20DRL%20Lecture%202/PPO2.png)
## Appendix 

**强化学习中策略优化方法（TRPO 和 PPO）的核心思想**：如何在更新策略时既提升性能，又避免因策略变化过大而导致训练不稳定。其中，**KL 散度（Kullback–Leibler Divergence）** 起到了“衡量策略变化程度”的作用。

下面我们从 **定义、来源、数学意义、为何使用** 四个方面详细解释 KL 散度在 TRPO/PPO 中的角色。

---

#### 一、KL 散度的严格数学定义

对于两个**概率分布** $ P $ 和 $ Q $ 定义在同一个可测空间上，且 $ P $ 关于 $ Q $ **绝对连续**（即 $ Q(x) = 0 \Rightarrow P(x) = 0 $），则 **KL 散度** 定义为：

$$
\mathrm{KL}(P \parallel Q) = \mathbb{E}_{x \sim P} \left[ \log \frac{P(x)}{Q(x)} \right] = \int P(x) \log \frac{P(x)}{Q(x)} \, dx \quad \text{(连续)}
$$
或
$$
\mathrm{KL}(P \parallel Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} \quad \text{(离散)}
$$

> 注意：KL 散度 **不是对称的**，即 $ \mathrm{KL}(P \parallel Q) \neq \mathrm{KL}(Q \parallel P) $，因此它不是严格意义上的“距离”，而是一种**信息差异度量**。

---

#### 二、在 TRPO / PPO 中的 KL 散度具体形式

在策略梯度方法中，策略是一个参数化的概率分布：
- $ \pi_\theta(a|s) $：当前策略（待更新）
- $ \pi_{\theta'}(a|s) $：旧策略（用于采样的策略）

那么，在 TRPO 和 PPO 中出现的 $ \mathrm{KL}(\theta, \theta') $ 实际上是 **状态-动作联合分布下的 KL 散度**，通常定义为：

$$
\mathrm{KL}(\theta \parallel \theta') = \mathbb{E}_{s \sim d^{\pi_{\theta'}}(s)} \left[ \mathrm{KL}\big( \pi_\theta(\cdot|s) \parallel \pi_{\theta'}(\cdot|s) \big) \right]
$$

其中：
- $ d^{\pi_{\theta'}}(s) $ 是策略 $ \pi_{\theta'} $ 下的状态访问分布（state visitation distribution），即在执行旧策略时访问状态 $ s $ 的频率。
- 内层 KL 是在每个状态 $ s $ 下，新旧策略在动作分布上的 KL 散度。

展开内层 KL：
$$
\mathrm{KL}\big( \pi_\theta(\cdot|s) \parallel \pi_{\theta'}(\cdot|s) \big) = \mathbb{E}_{a \sim \pi_\theta(\cdot|s)} \left[ \log \frac{\pi_\theta(a|s)}{\pi_{\theta'}(a|s)} \right]
$$

所以整体为：
$$
\mathrm{KL}(\theta \parallel \theta') = \mathbb{E}_{s \sim d^{\pi_{\theta'}},\, a \sim \pi_\theta(\cdot|s)} \left[ \log \frac{\pi_\theta(a|s)}{\pi_{\theta'}(a|s)} \right]
$$

> ⚠️ 注意：有些文献（尤其是早期 TRPO 论文）会使用 **反向 KL**（即 $ \mathrm{KL}(\pi_{\theta'} \parallel \pi_\theta) $），但现代实现（如 OpenAI 的 PPO）通常采用上述 **前向 KL（以新策略为期望）** 或近似平均。不过，在局部更新（小步长）下，两者差异不大。

---

#### 三、KL 散度的数学与信息论意义

1. **信息论解释**：
   - KL 散度衡量的是：**用分布 $ Q $ 来编码来自分布 $ P $ 的数据时，平均额外需要多少比特（或 nats）**。
   - 若 $ P = Q $，则 $ \mathrm{KL}(P \parallel Q) = 0 $；否则 $ > 0 $（Gibbs 不等式）。

2. **局部几何意义**：
   - 在参数空间中，KL 散度在 $ \theta' $ 附近可近似为 **Fisher 信息矩阵诱导的黎曼度量**：
     $$
     \mathrm{KL}(\theta \parallel \theta') \approx \frac{1}{2} (\theta - \theta')^\top F(\theta') (\theta - \theta')
     $$
     其中 $ F(\theta') = \mathbb{E}_{s,a}[\nabla \log \pi_{\theta'}(a|s) \nabla \log \pi_{\theta'}(a|s)^\top] $ 是 Fisher 信息矩阵。
   - 这正是 TRPO 使用**自然梯度**的理论基础：约束 KL 相当于在参数空间中限制更新步长在 Fisher 度量下的长度。

3. **为什么不用欧氏距离？**
   - 欧氏距离 $ \|\theta - \theta'\|^2 $ 忽略了策略函数的实际行为变化。
   - 例如，两个参数相差很大，但策略输出几乎相同（如 softmax 后概率不变），此时 KL ≈ 0，而欧氏距离可能很大。
   - KL 更能反映**策略行为的实际差异**。

---

#### 四、为什么 TRPO / PPO 要用 KL 散度？

##### 背景问题：策略梯度的“致命缺陷”

策略梯度方法基于如下近似：
$$
J(\theta) \approx J(\theta') + \mathbb{E}_{s,a \sim \pi_{\theta'}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta'}(a|s)} A^{\theta'}(s,a) \right]
$$
这个近似只在 $ \theta \approx \theta' $ 时成立。如果一步更新太大，新策略 $ \pi_\theta $ 与采样策略 $ \pi_{\theta'} $ 差异过大，会导致：
- 优势函数估计失效
- 训练剧烈震荡甚至崩溃

##### 解决方案：限制策略更新幅度

TRPO 和 PPO 的核心思想就是：**在保证性能提升的同时，限制新旧策略之间的差异**。

- **TRPO**：将 KL 散度作为**硬约束**（hard constraint）：
  $$
  \max_\theta J^{\theta'}(\theta) \quad \text{s.t.} \quad \mathrm{KL}(\theta \parallel \theta') \leq \delta
  $$
  通过共轭梯度法求解带约束的优化问题。

- **PPO**：将 KL 散度作为**软惩罚项**（soft penalty）加入目标函数：
  $$
  J_{\text{PPO}}(\theta) = J^{\theta'}(\theta) - \beta \cdot \mathrm{KL}(\theta \parallel \theta')
  $$
  （注：实际上 OpenAI 的 PPO 主要使用 **Clipped Surrogate Objective**，KL 惩罚版是其变种之一）

> ✅ 所以，KL 散度在这里的作用是：**量化新旧策略的行为差异，并防止策略更新过猛**。

---

#### 五、补充说明：KL 与重要性采样的关系

注意，策略梯度中的重要性权重 $ \frac{\pi_\theta(a|s)}{\pi_{\theta'}(a|s)} $ 与 KL 密切相关。事实上：

- 当 KL 很小时，重要性权重接近 1，方差低，估计稳定。
- 当 KL 很大时，重要性权重可能极端（如 0 或 ∞），导致高方差甚至偏差。

因此，控制 KL 也间接控制了重要性采样的稳定性。

---

#### 总结

| 项目 | 说明 |
|------|------|
| **KL 定义** | $ \mathrm{KL}(\pi_\theta \parallel \pi_{\theta'}) = \mathbb{E}_{s \sim d^{\pi_{\theta'}}, a \sim \pi_\theta} \left[ \log \frac{\pi_\theta(a|s)}{\pi_{\theta'}(a|s)} \right] $ |
| **来源** | 信息论（Kullback & Leibler, 1951）；在 RL 中由 Kakade & Langford (2002)、Schulman et al. (2015, TRPO) 引入策略优化 |
| **数学意义** | 衡量两个概率分布的非对称差异；局部等价于 Fisher 信息度量下的距离 |
| **为何使用** | 防止策略更新过大导致性能崩溃；比参数欧氏距离更能反映策略行为变化 |

> 📚 参考文献：
> - Schulman et al., *Trust Region Policy Optimization*, ICML 2015.
> - Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv 2017.
> - Kakade & Langford, *Approximately Optimal Approximate Reinforcement Learning*, ICML 2002.

