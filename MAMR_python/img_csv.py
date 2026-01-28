import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def img_to_csv(K, img_path, csv_path):
    img = Image.open(img_path).convert("L")
    img = img.resize((K, K))

    im = np.asarray(img, dtype=float)
    im = 1 - im / 255.0         
    im = np.maximum(im, 0)
    im /= im.sum()

    rows, cols = np.nonzero(im)
    values = im[rows, cols]

    data = np.column_stack((rows+1, cols+1, values))
    np.savetxt(csv_path, data, delimiter=",")


# for (i,j) in [(1,3),(4,8),(2,5),(6,7)]:
#     img_to_csv(64, rf"C:\Users\juann\Downloads\Mines\Scolarite\T2\MAM\MAMR_python\Data_inter\real_img\imagen_A_{i}.png", rf"C:\Users\juann\Downloads\Mines\Scolarite\T2\MAM\MAMR_python\Data_inter\imgc_sv\{i}.csv")
#     img_to_csv(64, rf"C:\Users\juann\Downloads\Mines\Scolarite\T2\MAM\MAMR_python\Data_inter\real_img\imagen_B_{j}.png", rf"C:\Users\juann\Downloads\Mines\Scolarite\T2\MAM\MAMR_python\Data_inter\imgc_sv\{j}.csv")


# for comb in ["1_3", "2_5", "4_8", "6_7"]:
#     img_to_csv(64, 
#                rf"C:\Users\juann\Downloads\Mines\Scolarite\T2\MAM\MAMR_python\Fig\originals\interpolacion\{comb}\barycenter_inter_non_regu_100.png",
#                  rf"C:\Users\juann\Downloads\Mines\Scolarite\T2\MAM\MAMR_python\Data_inter\imgc_sv\barycenters\{comb}_bary.csv")