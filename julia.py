from OpenGL.GL import *
import glfw
import numpy as np
import math
import cmath
from numba import cuda, float32, int32, complex64

width = 2**10
height = 2**10
ratio = height / width
center = np.array([0,0], dtype=np.float32)
xrange = np.float32(2)
yrange = xrange * ratio
c = np.complex64(-0.77623725+0.13852073j)

if not glfw.init():
    raise Exception("glfw cannot be initialized")

monitor = glfw.get_primary_monitor()

#window = glfw.create_window(2560,1080, "Fractals", monitor, None)
window = glfw.create_window(width,height, "Fractals", None, None)

if not window:
    glfw.terminate()
    raise Exception("glfw window cannot be created")

glfw.set_window_pos(window, 400,200)

# needs to be called to use openGL functions
glfw.make_context_current(window)

def key_callback(window, key, scancode, action, mods):
    global c, center, xrange, yrange
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)

    if key == glfw.KEY_A:
        c = np.complex64(cmath.rect(.7885, cmath.phase(c)-0.005))
        print('C', c)
    if key == glfw.KEY_D:
        c = np.complex64(cmath.rect(.7885, cmath.phase(c)+0.005))
        print('C', c)

    if key == glfw.KEY_W:
        xrange *= 1.1
        yrange = xrange * ratio
        print(xrange)
    if key == glfw.KEY_S:
        xrange /= 1.1
        yrange = xrange * ratio
        print(xrange)

    if key == glfw.KEY_UP:
        center[0] += xrange*0.01
        print(center)
    if key == glfw.KEY_DOWN:
        center[0] -= xrange*0.01
        print(center)
    if key == glfw.KEY_LEFT:
        center[1] -= yrange*0.01
        print(center)
    if key == glfw.KEY_RIGHT:
        center[1] += yrange*0.01
        print(center)

def scroll_callback(window, xoffset, yoffset):
    global xrange, yrange
    if yoffset > 0:
        xrange *= 1.1
        yrange = xrange * ratio
    if yoffset < 0:
        xrange /= 1.1
        yrange = xrange * ratio
    print(xrange)

def mouse_button_callback(window, button, action, mods):
    global xrange, yrange
    if button == glfw.MOUSE_BUTTON_LEFT:
        xrange *= 1.1
        yrange = xrange * ratio
    if button == glfw.MOUSE_BUTTON_RIGHT:
        xrange /= 1.1
        yrange = xrange * ratio
    print(xrange)

@cuda.jit
def kernel(A, x, y, c):
    i, k = cuda.grid(2)
    x_size = A.shape[0]
    y_size = A.shape[1]

    if i < x_size and k < y_size:
        count = int32(0)
        dist = float32(0)
        z = complex64(x[i] + y[k]*1j)
        while dist < 2 and count < 200:
            z = z**2+c
            count += 1
            dist = abs(z)
        A[i,k,0] += count

def render():
    global xrange, width, height, c, ratio, center, yrange

    x = np.linspace(center[0]-xrange/2, center[0]+xrange/2, width, dtype=np.float32)
    y = np.linspace(center[1]-yrange/2, center[1]+yrange/2, height, dtype=np.float32)

    pixels = np.zeros((x.shape[0], y.shape[0], 3), dtype=np.float32)

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(pixels.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(pixels.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    gpu_pixels = cuda.to_device(pixels)
    kernel[blockspergrid, threadsperblock](
        gpu_pixels,
        cuda.to_device(x), cuda.to_device(y), 
        c)
    result = gpu_pixels.copy_to_host()
    result = result - np.min(result[:,:,0])
    result = result/np.max(result)

    glDrawPixels(width, height, GL_RGB, GL_FLOAT, result)
    

glfw.set_key_callback(window, key_callback)
glfw.set_scroll_callback(window, scroll_callback)
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