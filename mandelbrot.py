from OpenGL.GL import *
import glfw
import numpy as np
import math
import cmath
from numba import cuda, float32, float64, int32, complex64, jit
from matplotlib import cm
import matplotlib.pyplot as plt

width = 2**10
height = 2**10
ratio = height / width
center = np.array([0,0], dtype=np.float64)
xrange = np.float64(2)
yrange = xrange * ratio

threadsperblock = (16, 16)
blockspergrid_x = math.ceil(width / threadsperblock[0])
blockspergrid_y = math.ceil(height / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
colormap = cm.get_cmap('hot')

if not glfw.init():
    raise Exception("glfw cannot be initialized")

monitor = glfw.get_primary_monitor()

#window = glfw.create_window(2560,1080, "Fractals", monitor, None)
window = glfw.create_window(width,height, "Fractals", None, None)

if not window:
    glfw.terminate()
    raise Exception("glfw window cannot be created")

glfw.set_window_pos(window, 0,0)

# needs to be called to use openGL functions
glfw.make_context_current(window)

def key_callback(window, key, scancode, action, mods):
    global center, xrange, yrange
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)

    if key == glfw.KEY_S:
        xrange *= 1.1
        yrange = xrange * ratio
    if key == glfw.KEY_W:
        xrange /= 1.1
        yrange = xrange * ratio

    if key == glfw.KEY_UP:
        center[0] += xrange*0.01
    if key == glfw.KEY_DOWN:
        center[0] -= xrange*0.01
    if key == glfw.KEY_LEFT:
        center[1] -= yrange*0.01
    if key == glfw.KEY_RIGHT:
        center[1] += yrange*0.01

def mouse_button_callback(window, button, action, mods):
    global xrange, yrange
    if button == glfw.MOUSE_BUTTON_LEFT:
        xrange *= 1.1
        yrange = xrange * ratio
    if button == glfw.MOUSE_BUTTON_RIGHT:
        xrange /= 1.1
        yrange = xrange * ratio

@cuda.jit
def kernel(A, x, y, lim):
    i, k = cuda.grid(2)
    x_size = A.shape[0]
    y_size = A.shape[1]

    if i < x_size and k < y_size:
        count = int32(0)
        z = complex64(0 + 0j)
        c = complex64(x[i] + y[k]*1j)
        converges = True
        while converges and count < lim:
            z = z**2+c
            count += 1
            converges = abs(z) < 2
        if count == lim:
            A[i,k] = -10
        else:
            A[i,k] = count

def color_mapper(A):
    return colormap(A)[...,:3]

def render():
    global xrange, width, height, c, ratio, center, yrange, blockspergrid, threadsperblock

    x = np.linspace(center[0]-xrange/2, center[0]+xrange/2, width, dtype=np.float64)
    y = np.linspace(center[1]-yrange/2, center[1]+yrange/2, height, dtype=np.float64)

    pixels = np.zeros((width, height), dtype=np.float32)

    gpu_pixels = cuda.to_device(pixels)
    kernel[blockspergrid, threadsperblock](
        gpu_pixels,
        cuda.to_device(x), cuda.to_device(y), math.ceil(100 / xrange))
    result = gpu_pixels.copy_to_host()
    result = (result - np.min(result))/np.max(result)

    glDrawPixels(width, height, GL_RGB, GL_FLOAT, color_mapper(result))
    

glfw.set_key_callback(window, key_callback)
glfw.set_mouse_button_callback(window, mouse_button_callback)


while not glfw.window_should_close(window):
    # Render here, e.g. using pyOpenGL
    glClear(GL_COLOR_BUFFER_BIT)
    render()
    
    # listens to inputs
    glfw.poll_events()

    # exchanges back and front buffer
    glfw.swap_buffers(window)

glfw.terminate()