---
noteID: 40e9e5b1-0f00-44b8-b81b-bdca672e7a65
tags:
  - Attention
---
SE блок **переосмысливает важность каждого канала** в feature map'е. Он _"сжимает"_ пространственную информацию (через Global Average Pooling), а потом _"возбуждает"_ важные каналы с помощью learnable весов.
![[Pasted image 20250612163832.png]]

Согласно первой схеме: 
$$z_c = \frac{1}{H \cdot W} \sum^{H}_{i=1}\sum^{W}_{j=1}X$$
$$s_c = MLP(ReLU(MLP(z_c)))$$
### $$\tilde X = \sigma(s_c) \cdot X$$
где, $z_c$ - результат глобального пуллинга (`GAP`), $s_c$ - выход полносвязных слоев $MLP(F)$ с функцией активации $ReLU(F)$.