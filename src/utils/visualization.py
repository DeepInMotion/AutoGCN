import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.animation import FFMpegWriter

EDGE_NTU = ((0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5),
            (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 0),
            (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
            (19, 18), (21, 22), (22, 7), (23, 24), (24, 11))

EDGE_Kinetics = ((0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
                 (11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15))

EDGE_Kinetics_3 = ((0, 1), (1, 2), (1, 5), (2, 8), (3, 2),
                 (4, 3), (5, 11), (5, 6), (6, 7),
                 (8, 9), (9, 10), (11, 12), (12, 13),
                 (14, 0), (15, 0), (16, 14), (17, 15))

EDGE_Kinetics_2 = ((0, 1), (2, 1), (3, 1), (1, 4), (9, 8),
                   (10, 9), (11, 10), (5, 8), (6, 5), (7, 6),
                   (4, 8), (12, 8), (16, 12), (17, 16), (18, 17),
                   (13, 12), (14, 13), (15, 14))


def vis(data, is_3d=True, pause=0.01, view=1, title='', gif=False, plotter=None):
    """

    :param data:
    :param is_3d:
    :param pause:
    :param view:
    :param title:
    :param gif:
    :param plotter:
    :return:
    """

    if data.shape[0] != 3:  # or data.shape[2] != 25:
        data = np.transpose(data, [3, 1, 2, 0])

    if data.shape[2] == 25:
        # NTU Edge
        edge = EDGE_NTU
    elif data.shape[2] == 18:
        # Kinetics Edge
        is_3d = False
        edge = EDGE_Kinetics
    else:
        raise NotImplemented('Data with {} edges are not supported'.format(data.shape[2]))

    matplotlib.use('macosx')
    C, T, V, M = data.shape  # C=3, T=300, V=25 or 18, M=2
    # print(C, T, V, M)
    plt.ion()
    fig = plt.figure()
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(elev=100, azim=-90)
    p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
    import sys
    from os import path
    sys.path.append(
        path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

    def plotter(write=None):
        pose = []
        for m in range(M):
            a = []
            for i in range(len(edge)):
                if is_3d:
                    a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                else:
                    a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
            pose.append(a)
        ax.axis([-view, view, -view, view])
        if is_3d:
            ax.set_zlim3d(-view, view)
        for t in range(T):
            for m in range(M):
                for i, (v1, v2) in enumerate(edge):
                    x1 = data[:2, t, v1, m]
                    x2 = data[:2, t, v2, m]
                    if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                        pose[m][i].set_xdata(data[0, t, [v1, v2], m])
                        pose[m][i].set_ydata(data[1, t, [v1, v2], m])
                        if is_3d:
                            pose[m][i].set_3d_properties(data[2, t, [v1, v2], m])
            fig.canvas.draw()
            if write is not None:
                write.grab_frame()
            # img.append(np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8'))
            # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.pause(pause)
        plt.close()
        plt.ioff()

    if gif:
        writer = FFMpegWriter(fps=25)
        plt.rcParams['animation.ffmpeg_path'] = '/opt/homebrew/bin/ffmpeg'
        with writer.saving(fig, "{}.gif".format(title), 100):
            plotter(writer)
        # images = plotter()
        # anim = animation.FuncAnimation(fig, images, frames=299, interval=50, blit=True)
        # writer = PillowWriter(fps=20)
        # anim.save("demo2.gif", writer='imagemagick')
        # plt.show()
    else:
        plotter(None)


def visualize_cp_skeleton(data):
    # TODO
    pass


def visualize_kinetics_skeleton(data):
    # TODO
    pass
