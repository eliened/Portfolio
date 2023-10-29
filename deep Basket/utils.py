import cv2
import os


def save_video_as_images(src, dst, name, rate):
    cap = cv2.VideoCapture(src)
    i = 1
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            if i % rate == 0:
                im = frame
                cv2.imwrite(f'{dst}/{name}_{i // rate}.jpg', im)
            i += 1
        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()
def get_cnt():
    imgs = os.listdir('custom_data/imgs') 
    imgs.remove('.DS_Store')
    if len(imgs) == 0:
        return 1
    imgs = [int(im[im.find('_')+1:im.rfind('_')]) for im in imgs]
    return max(*imgs)

def make_imgs():
    rate_dict = {'short': 10, 'mid': 35, 'long': 600}
    cnt = get_cnt()
    for tm in ['short', 'mid', 'long']:
        videos = os.listdir(f'custom_data/raw/videos/{tm}')
        videos.remove('.DS_Store')
        for video in videos:
            save_video_as_images(f'custom_data/raw/videos/{tm}/{video}', 'custom_data/imgs_gopro2', f'IMG_{cnt}',
                                 rate_dict[tm])
            cnt += 1
            print(f'images for {video} saved')

def change_class_yolo(path, new_cls):
    with open(path, "r") as f:
        filedata = f.read()
    with open(path, 'w') as f:
        f.write(str(new_cls) + filedata[1:])

def change_class_of_hoop_ds():
    for folder in ['valid', 'test', 'train']:
        path = f'custom_data/hoop_ds/{folder}/labels'
        files = os.listdir(path)

        for file in files:
            change_class_yolo(path + '/' + file, 1)

def add_hoop_label():

    files= os.listdir('custom_data/labels_gopro')

    for file in files:
        if 'IMG_18' in file:
            path = 'custom_data/labels_gopro/' + file
            with open(path, "r") as f:
                filedata = f.read()
            with open(path, 'w') as f:
                f.write(str(filedata) + '1 0.433594 0.282407 0.039062 0.055556')



import cv2 as cv

""" CODE POUR PRENDRE L'IMAGE
im = cv.imread('IMG_18_359.jpg')
s = '0.433594 0.282407 0.039062 0.055556'
arr = s.split()
arr = [float(x) for x in arr]
arr = [ (arr[0] - 0.5 * arr[2]) * 1920, (arr[1]- 0.5 * arr[3]) * 1080, (arr[0] + 0.5 * arr[2]) * 1920, (arr[1]+ 0.5 * arr[3]) * 1080]
arr = [int(x) for x in arr]
x1,y1,x2,y2 = arr
small_im = im[y1:y2, x1:x2, :]
cv.imwrite('larue.jpg', small_im)"""



def classify(dir):
    n = 0 #number of the try
    if not os.path.isdir('ball_in_hoop'):
        os.mkdir('ball_in_hoop')
        os.mkdir('ball_in_hoop/exp0')
        f = open('ball_in_hoop/n.txt', 'x')
        f.write('1')
        f.close()
    else:
        with open('ball_in_hoop/n.txt', 'r') as f:
            n = int(f.read())
        with open('ball_in_hoop/n.txt', 'w') as f:
            f.write(str(n + 1))
        os.mkdir(f'ball_in_hoop/exp{n}')
    os.mkdir(f'ball_in_hoop/exp{n}/in')
    os.mkdir(f'ball_in_hoop/exp{n}/out')


    rate = 3
    cap = cv2.VideoCapture(dir)
    i = 1
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            if i % rate == 0:
                cv.imshow('frame', frame)
                out = cv.waitKey(0) & 0xFF

                if out == ord('i'):
                    cv.imwrite(f'ball_in_hoop/exp{n}/in/img{i}.jpg', frame)
                if out == ord('o'):
                    cv.imwrite( f'ball_in_hoop/exp{n}/out/img{i}.jpg', frame)

            i += 1
        # Break the loop
        else:
            break

classify('custom_data/raw/videos/long/GOPR0635.MP4')