from OpenGL.GL import *
import glfw
import numpy as np
import math
import cmath
from numba import cuda, float32, int32, complex64
from matplotlib import cm

#draw states
REDRAW = 0
HOLD_1 = 1
REFINE = 2
HOLD_2 = 3

draw_state = REDRAW

width = 2**10
height = 2**10
ratio = height / width
center = np.array([0,0], dtype=np.float32)
xrange = np.float32(2)
yrange = xrange * ratio
c = np.array([-0.77623725, 0.15452073], dtype=np.float32)
colormap = cm.get_cmap("RdYlGn")

xmouse, ymouse = np.float32(0), np.float32(0)

threadsperblock = (16, 16)
blockspergrid_x = math.ceil(width / threadsperblock[0])
blockspergrid_y = math.ceil(height / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)


x = np.linspace(center[0]-xrange/2, center[0]+xrange/2, width, dtype=np.float32)
y = np.linspace(center[1]-yrange/2, center[1]+yrange/2, height, dtype=np.float32)
pixels = np.zeros((width, height), dtype=np.int32)
re_Z = np.zeros((width, height), dtype=np.float32)
im_Z = np.zeros((width, height), dtype=np.float32)

if not glfw.init():
    raise Exception("glfw cannot be initialized")

monitor = glfw.get_primary_monitor()

#window = glfw.create_window(2560,1080, "Fractals", monitor, None)
window = glfw.create_window(width,height, "Fractals", None, None)

if not window:
    glfw.terminate()
    raise Exception("glfw window cannot be created")

glfw.set_window_pos(window, 100,100)

# needs to be called to use openGL functions
glfw.make_context_current(window)

def key_callback(window, key, scancode, action, mods):
    global c, xrange, yrange, center, draw_state, HOLD_1
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)

    if key == glfw.KEY_A:
        draw_state = HOLD_1
    if key == glfw.KEY_D:
        draw_state = HOLD_1

    if key == glfw.KEY_W:
        draw_state = HOLD_1
        xrange /= 1.1
        yrange = xrange * ratio
    if key == glfw.KEY_S:
        draw_state = HOLD_1
        xrange *= 1.1
        yrange = xrange * ratio

    if key == glfw.KEY_UP:
        draw_state = HOLD_1
        center[0] += xrange*0.01
    if key == glfw.KEY_DOWN:
        draw_state = HOLD_1
        center[0] -= xrange*0.01
    if key == glfw.KEY_LEFT:
        draw_state = HOLD_1
        center[1] -= yrange*0.01
    if key == glfw.KEY_RIGHT:
        draw_state = HOLD_1
        center[1] += yrange*0.01

def cursor_position_fallback(window, xpos, ypos):
    global width, height
    #print(xpos-width/2, ypos-height/2)

@cuda.jit
def kernel(A, re_scale, im_scale, re_Z, im_Z, c):
    i, k = cuda.grid(2)
    x_size = A.shape[0]
    y_size = A.shape[1]

    threshold = int32(100)

    if i < x_size and k < y_size:
        count = int32(0)
        dist = float32(0)
        re_c, im_c = c[0], c[1]
        re, im = re_scale[i], im_scale[k]
        while dist < 2 and count < threshold:
            re_ = re*re - im*im + re_c
            im_ = (im + im)*re + im_c
            re = re_
            im = im_
            count += 1
            dist = math.sqrt(im*im + re*re)
        re_Z[i,k] = re
        im_Z[i,k] = im
        A[i,k] = count

@cuda.jit
def kernel_refine(A, re_Z, im_Z, c, start):
    i, k = cuda.grid(2)
    x_size = A.shape[0]
    y_size = A.shape[1]

    threshold = int32(100) + start

    if i < x_size and k < y_size:
        count = A[i,k]
        re_c, im_c = c[0], c[1]
        re, im = re_Z[i,k], im_Z[i,k]
        dist = math.sqrt(im*im + re*re)
        while dist < 2 and count < threshold:
            re_ = re*re - im*im + re_c
            im_ = (im + im)*re + im_c
            re = re_
            im = im_
            count += 1
            dist = math.sqrt(im*im + re*re)
        re_Z[i,k] = re
        im_Z[i,k] = im
        A[i,k] = count

def color_mapper(A):
    return colormap(A)[...,:3]

glfw.set_key_callback(window, key_callback)
glfw.set_cursor_pos_callback(window, cursor_position_fallback)

hold_count = 0
refine_level = 100
while not glfw.window_should_close(window):
    # Render here, e.g. using pyOpenGL
    if draw_state==REDRAW:
        refine_level = 100
        hold_count = 0
        x = np.linspace(center[0]-xrange/2, center[0]+xrange/2, width, dtype=np.float32)
        y = np.linspace(center[1]-yrange/2, center[1]+yrange/2, height, dtype=np.float32)

        gpu_pixels = cuda.to_device(pixels)
        kernel[blockspergrid, threadsperblock](
            gpu_pixels,
            cuda.to_device(x), cuda.to_device(y), 
            cuda.to_device(re_Z), cuda.to_device(im_Z),
            cuda.to_device(c))
        pixels = gpu_pixels.copy_to_host()  

        glDrawPixels(width, height, GL_RED, GL_FLOAT, (pixels - pixels.min())/pixels.max())
        print(refine_level, pixels.max())
        draw_state = HOLD_2
    elif draw_state==REFINE:
        hold_count = 0
        
        if refine_level < 10000:
            gpu_pixels = cuda.to_device(pixels)
            kernel_refine[blockspergrid, threadsperblock](
                gpu_pixels,
                cuda.to_device(re_Z), cuda.to_device(im_Z),
                cuda.to_device(c),
                refine_level)
            pixels = gpu_pixels.copy_to_host()
            refine_level += 100
            a = np.quantile(pixels, .95)
            a = np.where(pixels > a, a, pixels)
            glDrawPixels(width, height, GL_RGB, GL_FLOAT, colormap((a - np.min(a))/np.max(a))[...,:3])
            print(refine_level, a.max())
    elif draw_state==HOLD_1:
        refine_level = 100
        glDrawPixels(width, height, GL_RED, GL_FLOAT, pixels)
        if hold_count == 1:
            draw_state = REDRAW
            continue
        hold_count += 1
    elif draw_state==HOLD_2:
        refine_level = 100
        glDrawPixels(width, height, GL_RED, GL_FLOAT, pixels)
        if hold_count == 1:
            draw_state = REFINE
            continue
        hold_count += 1

    # listens to inputs
    glfw.poll_events()
    # exchanges back and front buffer
    glfw.swap_buffers(window)
    

glfw.terminate()