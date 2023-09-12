import matplotlib.pyplot as plt
import numpy as np
_click_started = False

def OnPress(event):
    global _click_started
    _click_started = True

def OnMove(event):
    global _click_started
    _click_started = False

def OnRelease(event, func, *args, **kwargs):
    global _click_started
    if not _click_started:
        return
    _click_started = False
    pt_xy = np.array((np.round(event.xdata), np.round(event.ydata))).astype(int)
    print('User clicked at location', pt_xy)
    func(pt_xy, *args, **kwargs)
    return True

def launch_clicky(arr, pnts, pltkwargs, func, *args, **kwargs):
    plt.figure()
    fig = plt.gcf()
    plt.imshow(arr.T, **pltkwargs)
    plt.colorbar()
    if pnts is not None:
        plt.scatter(pnts['x_loc'], pnts['y_loc'], color='red')
    cid_up = fig.canvas.mpl_connect('button_press_event', OnPress)
    cid_up = fig.canvas.mpl_connect('motion_notify_event', OnMove)
    cid_up = fig.canvas.mpl_connect('button_release_event', lambda event: OnRelease(event, func, *args, **kwargs))
    plt.show()
