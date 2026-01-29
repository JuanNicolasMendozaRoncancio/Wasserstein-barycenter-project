import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Sena
#===========================0
x = np.array([9.3
              , 6.97
              , 5.38
              , 4.46
              ,3.84
              ,2.56
              ,1.71
              ,1.88
              ,1.76
              ,2.69
              ,4
              ,4.81
              ,6.01
              ,7
              ,7.76
              ,9.36
              ,10.82
              ,11.56
              ,12.22
              ,12.97
              ,13.68
              ,14.22
              ,15.05
              ,15.78
              ,17.28
              ,19
              ,20
              ,20.79
              ,21.11
])
y = np.array([13.43,
              12.18,
              11.41,
              10.25,
              9.33,
              7.95,
              6.37,
              4.7,
              3.23,
              2.05,
              2,
              2.96,
              4.86,
              6,
              6.88,
              7.11,
              6.52,
              6.23,
              5.75,
              5.21,
              4.55,
              3.69,
              2.8,
              1.95,
              1.28,
              1.26,
              1.2,
              1,
              0.48

])

dx = np.diff(x)
dy = np.diff(y)
t = np.concatenate(([0], np.cumsum(np.sqrt(dx**2 + dy**2))))

sx = CubicSpline(t, x)
sy = CubicSpline(t, y)

tt = np.linspace(t[0], t[-1], 500)


## grid
#=============================
K = 64
xmin , xmax = min(x), max(x)
ymin , ymax = min(y), max(y)

xg = np.linspace(xmin, xmax, K)
yg = np.linspace(ymin, ymax, K)

tt = np.linspace(t[0], t[-1], 800)
river_x = sx(tt)
river_y = sy(tt)


## Mask
#============================
def seine_mask(K, sx, sy, t, width=0.01):

    X, Y = np.meshgrid(xg, yg)

    tt = np.linspace(t[0], t[-1], 800)
    rx = sx(tt)
    ry = sy(tt)

    mask = np.ones((K, K), dtype=float)

    for i in range(K):
        for j in range(K):
            dx = rx - X[i, j]
            dy = ry - Y[i, j]
            dist = np.min(np.sqrt(dx**2 + dy**2))
            if dist < width:
                mask[i, j] = 0.0

    return mask

mask = seine_mask(K, sx, sy, t, width=0.3)



#Density population
#============================
def population_DoG(K,
                   c_inner=(14.5, 8),
                   c_outer=((min(x)+max(x))/2, (min(y)+max(y))/2),
                   sigma_outer=2.5,
                   sigma_inner=6,
                   lambda_inner=0.8,  # Aumentado para que la resta sea más visible
                   mask=None):
    
    xgrid = np.linspace(min(x), max(x), K)
    ygrid = np.linspace(min(y), max(y), K)
    X, Y = np.meshgrid(xgrid, ygrid)
    
    # Gaussiana exterior (más ancha)
    R2_outer = (X - c_outer[0])**2 + (Y - c_outer[1])**2
    gauss_outer = np.exp(-R2_outer / (2 * sigma_outer**2))
    
    # Gaussiana interior (más estrecha, en posición diferente)
    R2_inner = (X - c_inner[0])**2 + (Y - c_inner[1])**2
    gauss_inner = np.exp(-R2_inner / (2 * sigma_inner**2))
    
    # Diferencia de Gaussianas
    pop = gauss_inner - lambda_inner * gauss_outer
    
    # Asegurar valores no negativos
    pop = np.maximum(pop, 0)
    
    # Aplicar máscara ANTES de normalizar
    if mask is not None:
        pop *= mask
    
    # Normalizar solo después de aplicar la máscara
    if pop.sum() > 0:
        pop /= pop.sum()
    
    return pop, gauss_outer, gauss_inner

pop, gauss_outer, gauss_inner = population_DoG(K, mask=mask)


#Providers
#============================
def generate_provider(K, pop_density, mask,
                      n_centers=5,
                      sigma_range=(1, 3),
                      alpha=0.5):

    x_grid = np.linspace(min(x), max(x), K)
    y_grid = np.linspace(min(y), max(y), K)
    X, Y = np.meshgrid(x_grid, y_grid)

    g = np.zeros((K, K))
    for _ in range(n_centers):
        cx= np.random.uniform(min(x), max(x))
        cy= np.random.uniform(min(y), max(y))
        sigma = np.random.uniform(*sigma_range)
        weight = np.random.uniform(0.5, 1.5)

        g += weight * np.exp(
            -((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2)
        )

    g *= mask
    g /= g.sum()

    q = alpha * pop_density + (1 - alpha) * g
    q *= mask
    q /= q.sum()
    
    umbr = np.percentile(q, 35)
    q = np.where(q >= umbr, q, 0.0)
    q /= q.sum()
    return q


alphas = [np.random.uniform(0, 0.90) for _ in range(20)]
providers = [
    generate_provider(K, pop, mask, alpha=a)
    for a in alphas
]

# for provieder in providers:
#     fig = plt.figure(figsize=(8, 6))
#     plt.imshow(provieder, origin='lower',
#             extent=(min(x), max(x), min(y), max(y)),
#             cmap='viridis')
#     plt.title("Population Density")
#     plt.colorbar(label='Density')
#     plt.show()


output_dir = r"C:\Users\juann\Downloads\Mines\Scolarite\T2\MAM\MAMR_python\Data_app_4_5\real_imgs"
os.makedirs(output_dir, exist_ok=True)

for i, q in enumerate(providers):
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.imshow(q, origin='lower',
              extent=(min(x), max(x), min(y), max(y)),
              cmap='viridis')

    ax.axis('off')

    filename = f"provider_{i:02d}.png"
    plt.savefig(
        os.path.join(output_dir, filename),
        dpi=300,
        bbox_inches='tight',
        pad_inches=0
    )

    plt.close(fig)
