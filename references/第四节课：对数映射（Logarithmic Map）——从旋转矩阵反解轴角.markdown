### 第四节课：对数映射（Logarithmic Map）——从旋转矩阵反解轴角

罗德里格斯公式（指数映射 $\exp$）实现了从向量到矩阵的转换：


$$\text{轴角 } \theta \mathbf{n} \in \mathbb{R}^3 \xrightarrow{\text{罗德里格斯公式 } \exp} \text{旋转矩阵 } \mathbf{R} \in SO(3)$$

现在我们需要解决它的逆问题——**对数映射（$\log$）**：
如果已知一个旋转矩阵 $\mathbf{R}$（例如机械臂当前读出的姿态，或专家示范的目标姿态），**如何反求出它是绕哪根轴 $\mathbf{n}$、转过了多大的角度 $\theta$？**

---

#### 1. 第一步：求转角 $\theta$（利用矩阵的“迹”）

矩阵的迹（Trace，记作 $\mathrm{tr}$）是主对角线三个数字相加：


$$\mathrm{tr}(\mathbf{R}) = R_{11} + R_{22} + R_{33}$$

对罗德里格斯公式两边同时求迹：


$$\mathbf{R} = \mathbf{I} + (\sin\theta)\hat{\mathbf{n}} + (1 - \cos\theta)\hat{\mathbf{n}}^2$$

逐项看对角线之和：

1. **$\mathrm{tr}(\mathbf{I}) = 1 + 1 + 1 = 3$**
2. **$\mathrm{tr}(\hat{\mathbf{n}}) = 0$**（因为反对称矩阵对角线恒为 0）
3. **$\mathrm{tr}(\hat{\mathbf{n}}^2)$ 是多少？**
由于 $\hat{\mathbf{n}}^2 = \mathbf{n}\mathbf{n}^\top - \mathbf{I}$：
* $\mathrm{tr}(\mathbf{n}\mathbf{n}^\top) = n_1^2 + n_2^2 + n_3^2 = \Vert{}\mathbf{n}\Vert{}^2 = 1$（单位向量模长为 1）
* $\mathrm{tr}(\mathbf{I}) = 3$
* 所以 $\mathrm{tr}(\hat{\mathbf{n}}^2) = 1 - 3 = -2$



把这三项代回等式：


$$\begin{aligned} \mathrm{tr}(\mathbf{R}) &= 3 + (\sin\theta) \cdot 0 + (1 - \cos\theta) \cdot (-2) \\ &= 3 - 2 + 2\cos\theta \\ &= 1 + 2\cos\theta \end{aligned}$$

移项整理，直接得到计算旋转角的公式：


$$\cos\theta = \frac{\mathrm{tr}(\mathbf{R}) - 1}{2} \implies \theta = \arccos\left( \frac{\mathrm{tr}(\mathbf{R}) - 1}{2} \right)$$

无论旋转矩阵多么复杂，**把对角线三个数字加起来，代入反余弦，就能直接算出旋转了多少弧度**。

---

#### 2. 第二步：求转轴 $\mathbf{n}$（利用矩阵与其转置的差）

现在已知了角度 $\theta$，如何求单位转轴 $\mathbf{n}$？

写出旋转矩阵 $\mathbf{R}$ 及其转置 $\mathbf{R}^\top$：

* $\mathbf{R} = \mathbf{I} + (\sin\theta)\hat{\mathbf{n}} + (1 - \cos\theta)\hat{\mathbf{n}}^2$
* $\mathbf{R}^\top = \mathbf{I}^\top + (\sin\theta)\hat{\mathbf{n}}^\top + (1 - \cos\theta)(\hat{\mathbf{n}}^2)^\top$

根据反对称矩阵的性质：$\hat{\mathbf{n}}^\top = -\hat{\mathbf{n}}$，以及 $(\hat{\mathbf{n}}^2)^\top = (-\hat{\mathbf{n}})(-\hat{\mathbf{n}}) = \hat{\mathbf{n}}^2$：


$$\mathbf{R}^\top = \mathbf{I} - (\sin\theta)\hat{\mathbf{n}} + (1 - \cos\theta)\hat{\mathbf{n}}^2$$

用上面两式直接相减（$\mathbf{R} - \mathbf{R}^\top$）：

* 单位矩阵 $\mathbf{I} - \mathbf{I} = \mathbf{0}$
* 平方项相减也直接抵消为 $\mathbf{0}$
* 只剩下中间的一阶项：

$$\mathbf{R} - \mathbf{R}^\top = 2(\sin\theta)\hat{\mathbf{n}}$$



移项整理：


$$\hat{\mathbf{n}} = \frac{\mathbf{R} - \mathbf{R}^\top}{2\sin\theta}$$

展开这个矩阵形式：


$$\begin{bmatrix} 0 & -n_3 & n_2 \\ n_3 & 0 & -n_1 \\ -n_2 & n_1 & 0 \end{bmatrix} = \frac{1}{2\sin\theta} \begin{bmatrix} 0 & R_{12}-R_{21} & R_{13}-R_{31} \\ R_{21}-R_{12} & 0 & R_{23}-R_{32} \\ R_{31}-R_{13} & R_{32}-R_{23} & 0 \end{bmatrix}$$

把帽子“摘下”，对应位置的三维转轴向量分量就是：


$$\mathbf{n} = \frac{1}{2\sin\theta} \begin{bmatrix} R_{32} - R_{23} \\ R_{13} - R_{31} \\ R_{21} - R_{12} \end{bmatrix}$$

最后，李代数向量就是转角乘转轴：


$$\boldsymbol{\omega} = \theta \mathbf{n} \in \mathbb{R}^3$$

---

#### 3. 几何直觉总结

* **对角线（迹）**：反映的是各轴在自身方向上的投影残留，总和直接给出了**旋转角 $\theta$ 的大小**；
* **非对角线相减（反对称部分）**：抵消了所有平面对称分量，精确提取出了围绕哪根轴旋转的**转轴空间指向 $\mathbf{n}$**。

---

**检查点：**
看一看利用“矩阵的迹求 $\theta$”和“矩阵减转置求 $\mathbf{n}$”的推导过程，每一步代数计算是否顺畅？确认理解后，我们继续探讨其中的一个工程边界情况（当角度 $\theta \approx 0$ 时分母为零怎么办），或者直接进入流形上的测地线插值。