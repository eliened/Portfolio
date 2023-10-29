#################################################################
# FILE : image_editor.py
# WRITER : your_name , your_login , your_id
# EXERCISE : intro2cs ex5 2022-2023
# DESCRIPTION: A simple program that...
# STUDENTS I DISCUSSED THE EXERCISE WITH: Bugs Bunny, b_bunny.
#								 	      Daffy Duck, duck_daffy.
# WEB PAGES I USED: www.looneytunes.com/lola_bunny
# NOTES: ...
#################################################################

##############################################################################
#                                   Imports                                  #
##############################################################################
import math
import sys
from math import floor

from ex5_helper import *
from typing import Optional


##############################################################################
#                                  Functions                                 #
##############################################################################
GRAY_SCALE = 'G'


def separate_channels(image: ColoredImage) -> List[SingleChannelImage]:
    res = []
    num_channels = len(image[0][0])
    for i in range(num_channels):
        channel_lines = []
        for line in image:
            channel_line = []
            for channels in line:
                channel_line.append(channels[i])
            channel_lines.append(channel_line)
        res.append(channel_lines)
    return res


def combine_channels(channels: List[SingleChannelImage]) -> ColoredImage:
    num_channels = len(channels)
    num_lines = len(channels[0])
    num_cols = len(channels[0][0])

    image = []
    for i in range(num_lines):
        image.append([])
        for _ in range(num_cols):
            image[i].append([])

    for c in range(num_channels):
        for i in range(num_lines):
            for j in range(num_cols):
                image[i][j].append(channels[c][i][j])

    return image


def RGB2grayscale(colored_image: ColoredImage) -> SingleChannelImage:
    def channel2grayscale(channel):
        return round(channel[0] * 0.299 + channel[1] * 0.587 + channel[2] * 0.114)

    gray_image = []
    for line in colored_image:
        gray_line = []
        for channel in line:
            gray_line.append(channel2grayscale(channel))
        gray_image.append(gray_line)
    return gray_image


def blur_kernel(size: int) -> Kernel:
    return [[1 / (size ** 2) for _ in range(size)] for _ in range(size)]


'''def apply_kernel(image: SingleChannelImage, kernel: Kernel) -> SingleChannelImage:
    side = len(kernel) // 2
    nrows = len(image)
    ncolumns = len(image[0])
    rows = []
    for _ in range(ncolumns):
        rows.append(0)
    new_image = [[rows] * nrows]
    for i_image in range(len(image)):
        for j_image in range(len(image)):
            for i_kernel in range(len(kernel)):
                for j_kernel in range(len(kernel)):
                    i_transposed = i_kernel - side
                    j_transposed = j_kernel - side
                    if i_image + i_transposed < 0:
                        if i_image + i_transposed >= len(image):
                            if j_image + j_transposed < 0:
                                if j_image + j_transposed >= len(image[0]):
                                    new_image[i_image][j_image] += image[i_image][j_image] * kernel[i_kernel][j_kernel]
                    else:
                        new_image[i_image][j_image] += image[i_image + i_kernel][j_image + j_kernel] * kernel[i_kernel][
                            j_kernel]

'''


def apply_kernel(image: SingleChannelImage, kernel: Kernel) -> SingleChannelImage:
    num_lines = len(image)
    num_cols = len(image[0])
    k_size = len(kernel)
    half_k_size = k_size // 2

    def conv_op(im, k, i, j):
        sum = 0.0
        for i_im, i_k in zip(range(i - half_k_size, i + half_k_size + 1), range(0, k_size)):

            for j_im, j_k in zip(range(j - half_k_size, j + half_k_size + 1), range(0, k_size)):
                if 0 <= j_im < num_cols and 0 <= i_im < num_lines:
                    sum += im[i_im][j_im] * k[i_k][j_k]
                else:
                    sum += im[i][j] * k[i_k][j_k]
        sum = round(sum)

        if sum > 255:
            return 255
        if sum < 0:
            return 0
        return sum

    res = []

    for i in range(num_lines):
        line = []
        for j in range(num_cols):
            line.append(conv_op(image, kernel, i, j))
        res.append(line)
    return res


def bilinear_interpolation(image: SingleChannelImage, y: float, x: float) -> int:
    h = len(image)
    w = len(image[0])


    x_floor = math.floor(x)
    x_ceil = min(w - 1, math.ceil(x))
    y_floor = math.floor(y)
    y_ceil = min(h - 1, math.ceil(y))
    #print(f'h: {h}, w:{w}, x_ceil = {x_ceil}, x_floor: {x_floor}, y_ceil: {y_ceil}, y_floor: {y_floor}, new_x: {x}, new_y: {y}')

    if (x_ceil == x_floor) and (y_ceil == y_floor):
        q = image[int(x_ceil)][int(x_ceil)]

    elif x_ceil == x_floor:
        q1 = image[ int(y_floor)][int(x_ceil)]
        q2 = image[ int(y_ceil)][int(x_ceil)]

        #print(f'q1: {q1}, q2: {q2}')

        q = q1 * (y_ceil - y) + q2 * (y - y_floor)
    elif y_ceil == y_floor:
        q1 = image[int(y_ceil)][int(x_floor)]
        q2 = image[int(y_ceil)][int(x_ceil)]
        #print(f'q1: {q1}, q2: {q2}')
        q = q1 * (x_ceil - x) + q2 * (x - x_floor)
    else:
        a = image[y_floor][x_floor]
        b = image[y_ceil][x_floor]
        c = image[y_floor][x_ceil]
        d = image[y_ceil][x_ceil]
        #print(f'a: {a}, b: {b}, c: {c}, d: {d} ')
        q1 = a * (x_ceil - x) + b * (x - x_floor)
        q2 = c * (x_ceil - x) + d * (x - x_floor)
        q = q1 * (y_ceil - y) + q2 * (y - y_floor)


    return math.ceil(q)







def resize(image: SingleChannelImage, new_height: int, new_width: int) -> SingleChannelImage:
    old_h = len(image)
    old_w = len(image[0])

    w_scale_factor = old_w / new_width
    h_scale_factor = old_h / new_height

    new_im = []
    mat_coord = []
    for i in range(0,new_height):
        line = []
        c_line = []
        for j in range(0, new_width):
            y =i  * h_scale_factor
            x =j  * w_scale_factor
            c_line.append([round(y,2),round(x,2)])
            line.append(bilinear_interpolation(image, y,x))
        new_im.append(line)
        mat_coord.append(c_line)

    new_im[0][0] = image[0][0]
    new_im[0][-1] = image[0][-1]
    new_im[-1][0] = image[-1][0]
    new_im[-1][-1] = image[-1][-1]
    #print(mat_coord)
    #print(new_im)
    return new_im
def rotate_90(image: Image, direction: str) -> Image:
    num_lines = len(image)
    num_cols = len(image[0])
    if direction == 'R':
        new_image = list(reversed(image))

    if direction == 'L':
        new_image = []
        for line in image:
            new_image.append(list(reversed(line)))
    res = []
    for i in range(num_cols):
        res.append([])

    for i in range(num_lines):
        for j in range(num_cols):
            res[j].append(new_image[i][j])

    return res


def get_edges(image: SingleChannelImage, blur_size: int, block_size: int, c: float) -> SingleChannelImage:
    blurred_image = apply_kernel(image, blur_kernel(blur_size))
    half_k_size = block_size // 2
    num_lines = len(image)
    num_cols = len(image[0])
    k_size = block_size

    def avg_op(im, i, j):
        vals = []
        for i_im, i_k in zip(range(i - half_k_size, i + half_k_size + 1), range(0, k_size)):

            for j_im, j_k in zip(range(j - half_k_size, j + half_k_size + 1), range(0, k_size)):
                if 0 <= j_im < num_cols and 0 <= i_im < num_lines:
                    vals.append(im[i_im][j_im])
                else:
                    vals.append(im[i][j])
        return sum(vals)/len(vals)
    new_img = []

    for i in range(len(blurred_image)):
        new_line = []
        for j in range(len(blurred_image[i])):
            t = avg_op(blurred_image, i,j) - c
            if blurred_image[i][j] < t:
                new_line.append(0)
            else:
                new_line.append(255)
        new_img.append(new_line)
    return new_img






def quantize(image: SingleChannelImage, N: int) -> SingleChannelImage:
    return [[round(math.floor(image[i][j] * (N / 256)) * (255 / (N - 1))) for j in range(len(image[0]))] for i in
            range(len(image))]


def quantize_colored_image(image: ColoredImage, N: int) -> ColoredImage:
    channels = separate_channels(image)
    quantized_channels = [quantize(c, N) for c in channels]
    return combine_channels(quantized_channels)

def type_image(img):
    if type(img[0][0]) == type(2):
        return GRAYSCALE_CODE
    else:
        return RGB_CODE

if __name__== "__main__":
    img_path = sys.argv[1]
    img = load_image(img_path)
    keep_running = True
    while keep_running:

        x = input('Choose operation: ')
        while x not in [str(i) for i in range(1,9)]:
            print('Invalid op')
            x = input('Choose operation: ')

        # Convert to Grayscale
        if x == '1':
            if type_image(img) == RGB_CODE:
                img = RGB2grayscale(img)
                print('Successfully converted to grayscale')
            else:
                print('Image already in grayscale')

        # Blur the image
        elif x == '2':
            blur_size = input('Choose blur size: ')
            if not blur_size.isnumeric() and int(blur_size) % 2 == 0:
                print('Invalid Blur Size')
            else:
                blur_size = int(blur_size)
                k = blur_kernel(blur_size)
                if type_image(img) == GRAYSCALE_CODE:
                    img = apply_kernel(img, k)
                else:
                    img = combine_channels([apply_kernel(channel, k) for channel in separate_channels(img)])
                print('Successfully blurred')

        # Resize
        elif x == '3':
            params = input('Chose new height and width (format H,W): ')
            h,w = params.split(',')
            if h.isnumeric() and int(h) > 0 and w.isnumeric() and int(w) > 0:
                h = int(h)
                w = int(w)
                if type_image(img) == GRAYSCALE_CODE:
                    img = resize(img, h,w)
                else:
                    img = combine_channels([resize(channel,h,w) for channel in separate_channels(img)])
                print(f'Successfully reshaped to {h}x{w}')
            else:
                print('Invalid input')


        # Rotate the image:
        elif x == '4':
            rotation = input('Chose rotation (R for right or L for left): ')
            if rotation == 'R':
                img = rotate_90(img,rotation)
                print('Successfully rotated the image 90° right')
            elif rotation == 'L':
                img = rotate_90(img, rotation)
                print('Successfully rotated the image 90° left')
            else:
                print(f'Invalid rotation {rotation} is not R or L')

        # Edge detection
        elif x == '5':
            params = input('Chose the params (format: blur_size, block_size,block_size,c): ')
            blur_size, block_size, c = params.split(',')

            block_ok = block_size.isnumeric and int(block_size) % 2 == 1 and  int(block_size) > 0
            blur_ok = blur_size.isnumeric and int(blur_size) % 2 == 1 and int(blur_size) > 0
            c_ok = c.replace('.','').isnumeric() and float(c) >= 0

            if block_ok and blur_ok and c_ok:

                block = int(block_size)
                blur = int(blur_size)
                c = float(c)

                if type_image(img) == GRAYSCALE_CODE:
                    img = get_edges(img,blur, block,c)
                else:
                    img = get_edges(RGB2grayscale(img),blur, block,c)

                print('Edges Successfully detected')
            else:
                print('Invalid input')


        # Quantization
        elif x == '6':
            N = input('Chose the number of tints: ')

            if N.isnumeric() and 2 <= int(N) <= 255:
                N = int(N)
                if type_image(img) == RGB_CODE:
                    img = quantize_colored_image(img, N)
                else:
                    img = quantize(img, N)

                print(f'Successfully quantized the image to {N} colors')
            else:
                print('Invalid input')

        elif x == '7':
            show_image(img)
        elif x == '8':
            save_image(img,'my_'+img_path)
            print('Successfully saved the image')
            break



