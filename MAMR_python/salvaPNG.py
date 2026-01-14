import matplotlib.pyplot as plt

def salvaPNG(fig=None, filename='Fig.png'):
    if fig is None:
        fig = plt.gcf()

    width, height = fig.get_size_inches()

    fig.savefig(
        filename,
        dpi=fig.dpi,
        bbox_inches='tight'
    )
