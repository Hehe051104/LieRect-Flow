## $SO(3)$ 与 $SE(3)$ 核心参数对照表

| 对比维度 | 特殊正交群 $SO(3)$ | 特殊欧氏群 $SE(3)$ |
| :--- | :--- | :--- |
| **物理含义** | **纯旋转**（原地转角度） | **刚体变换**（位置移动 + 角度旋转） |
| **自由度** | 3 自由度（3-DOF） | 6 自由度（6-DOF：3 平移 + 3 旋转） |
| **群里的元素形式** | $3 \times 3$ 旋转矩阵 $\mathbf{R}$ | $4 \times 4$ 齐次变换矩阵 $\mathbf{T} = \begin{bmatrix} \mathbf{R} & \mathbf{p} \\ \mathbf{0}^{\mathrm{T}} & 1 \end{bmatrix}$ |
| **李代数速度形式** | 3 维角速度 $\boldsymbol{\omega} \in \mathfrak{so}(3)$ | 6 维运动旋量（Twist）$\boldsymbol{\xi} = \begin{bmatrix} \boldsymbol{\omega} \\ \mathbf{v} \end{bmatrix} \in \mathfrak{se}(3)$ |
| **生活比喻** | 站在原地转头看不同方向 | 走到桌子另一边，再低头看水杯 |