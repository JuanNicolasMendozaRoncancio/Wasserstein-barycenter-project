import matplotlib.pyplot as plt
import os

def salvaPNG(fig=None, filename='Fig.png', outputFolder='./'):

    if fig is None:
        fig = plt.gcf()

    width, height = fig.get_size_inches()
    os.makedirs(outputFolder, exist_ok=True)

    filepath = os.path.join(outputFolder, filename)

    fig.savefig(
        filepath,
        dpi=fig.dpi,
        bbox_inches='tight'
    )
