import os
import glob
import cv2


def mk_avi(files, fname, dir):

    files = sorted(files)

    images = []
    for file in files:
        images.append(cv2.imread(file))
    dim = images[0].shape
    framesize = (dim[1], dim[0])

    filename = os.path.join(dir, fname+'.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid = cv2.VideoWriter(filename, fourcc, 1, framesize)


    for im in images:
        #plt.imshow(im)
        #plt.show()
        vid.write(im)

    vid.release()


if __name__ == "__main__":
    im_dir = "/local-scratch/users/aplourde/snow/full_scene/TSX_SM39_D/"
    files = glob.glob(dir + '*.png')

    mk_avi(files, 'swe', im_dir)