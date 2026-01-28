import os
import random
import numpy as np
from PIL import Image
from torchvision import datasets, transforms

def extraer_dos_mnist_distintos(
    carpeta_destino,
    size=64,
    train=True
):
    os.makedirs(carpeta_destino, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

    mnist = datasets.MNIST(
        root="./mnist_data",
        train=train,
        download=True,
        transform=transform
    )

    indices_por_digito = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(mnist):
        indices_por_digito[label].append(idx)

    d1, d2 = 6,7 


    idx1 = random.choice(indices_por_digito[d1])
    idx2 = random.choice(indices_por_digito[d2])

    img1, _ = mnist[idx1]
    img2, _ = mnist[idx2]

    arr1 = img1.squeeze().numpy()
    arr2 = img2.squeeze().numpy()

    arr1 = 1.0 - arr1
    arr2 = 1.0 - arr2

    img1 = Image.fromarray((arr1 * 255).astype(np.uint8))
    img2 = Image.fromarray((arr2 * 255).astype(np.uint8))

    img1.save(os.path.join(carpeta_destino, f"imagen_A_{d1}.png"))
    img2.save(os.path.join(carpeta_destino, f"imagen_B_{d2}.png"))

    print(f"Imágenes seleccionadas: dígitos {d1} y {d2}")

carpeta_destino = r"C:\Users\juann\Downloads\Mines\Scolarite\T2\MAM\MAMR_python\Data_inter\real_img"

extraer_dos_mnist_distintos(carpeta_destino, size=64)
