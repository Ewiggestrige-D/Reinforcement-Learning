# Part 2 DRL Lecture 5 Q-learning (Continuous Action)
[DRL Lecture 5: Q-learning (Continuous Action)](https://www.youtube.com/watch?v=tnPVcec22cg)

## Q-learning for Continuous Action
- Continous Action
  - auto-driving
  - robotics
- Why "**Continous**" would cause problems?
  - if actions are concrete, the choices of actions are **Finite**, so we can choose different actions for *optimization*
  - if actions are continuous, the choices of actions are **INFinite**


### Solution 1
- Sample a set of actions: ${a_1,a_2,\dots,a_N}$
- See which action can obtain the largest $Q$ value (not very accurate)


### Solution 2
- Using **gradient ascent** to solve the optimization problem.
- Issue:
  - local minimal
  - calculate gradient **EVERY** time we need to choose the actions

### Solution 3
- Design a **network** to make the optimization easy.
![Network](/Hung-yi%20Lee/img/Part%202%20DRL%20Lecture%205/Network.png)

其中$\mu(s) = arg\max_{a}Q(s,a)$,即target function.

同时需要注意，$\Sigma(s)$ matrix 需要经过一些处理使得其变得正定，保证
$$V(s,a) = -\left(q- \mu(s) \right)^{T} \Sigma(s) \left(q- \mu(s) \right) +V(s)
$$
中$-\left(q- \mu(s) \right)^{T} \Sigma(s) \left(q- \mu(s) \right)$为负，来进行action 到 target function的regression 优化.

### Solution 4 Do not use Q-learning