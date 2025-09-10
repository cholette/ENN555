# %% 

import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x)**3
plt.plot(x, y)
plt.show()