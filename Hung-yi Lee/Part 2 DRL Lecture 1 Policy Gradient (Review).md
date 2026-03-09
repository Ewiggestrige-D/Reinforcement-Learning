# DRL Lecture 1: Policy Gradient (Review)
[DRL Lecture 1: Policy Gradient (Review)](https://www.youtube.com/watch?v=z95ZYgPgXOY)

## Proximal Policy Optimization (PPO)

### Review
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


#### Tip 1：**Add a baseline** 
为了避免$R(\tau)$都是正数，而sampling次数有限导致的sample并非全部都是最优解,可以手动减掉一个baseline值（hyper-parameter），来保证策略会保持在一个最低优化的策略上
$$\nabla \overline{R_{\theta}} 
  = \frac{1}{N}\sum_{n=1}^{N}  \sum_{t=1}^{T} (R(\tau)-b)\nabla \log p(a_t|s_t,\theta)
$$
  
  
#### Tip 2：**Asign suitable credit**

对于某一个timestep $t$的action $a_t$,其所造成的影响（reward）不应该由整个episode的reward累加，应该计算从时间$t$开始之后的所有action造成的reward累加，即
$$
\begin{aligned}
    & \nabla \overline{R_{\theta}} 
  = \frac{1}{N}\sum_{n=1}^{N}  \sum_{t=1}^{T} (R(\tau)-b)\nabla \log p(a_t|s_t,\theta) \\
 \rightarrow & \nabla \overline{R_{\theta,a_t}} =  \frac{1}{N}\sum_{n=1}^{N}  \sum_{t=1}^{T} (\sum_{t'=t}^{T}r_{t'}^{n}-b)\nabla \log p(a_t|s_t,\theta) \\
\end{aligned}
$$
 
此刻考虑时间因素，即对于timestep $t$之后的时刻$t'$间距越远，造成的影响$\gamma^{t'-t} < 1$越小，即
$$
\begin{aligned}
    & \nabla \overline{R_{\theta,a_t}} =  \frac{1}{N}\sum_{n=1}^{N}  \sum_{t=1}^{T} (\sum_{t'=t}^{T}r_{t'}^{n}-b)\nabla \log p(a_t|s_t,\theta) \\
  \rightarrow & \nabla \overline{R_{\theta,a_t}} =  \frac{1}{N}\sum_{n=1}^{N}  \sum_{t=1}^{T} (\sum_{t'=t}^{T}\gamma^{t'-t}\cdot r_{t'}^{n}-b)\nabla \log p(a_t|s_t,\theta) \\
\end{aligned}
$$
对于其中部分，做如下定义
Define **Advantage Function**
$$A^{\theta}(s_t,a_t) = \sum_{t'=t}^{T}\gamma^{t'-t}\cdot r_{t'}^{n}-b$$
 where $\theta$ labels the ***chosen model/strategy that interacts with the environment***