from OpenGL.GL import *
import glfw
import numpy as np
import math
import cmath
from numba import cuda, float32, int32, complex64
from matplotlib import cm

width = 2**10
height = 2**10
ratio = height / width
center = np.array([0,0], dtype=np.float32)
xrange = np.float32(2)
yrange = xrange * ratio
c = np.complex64(-0.77623725+0.15452073j)
colormap = cm.get_cmap("RdYlGn")

xmouse, ymouse = np.float32(0), np.float32(0)

threadsperblock = (16, 16)
blockspergrid_x = math.ceil(width / threadsperblock[0])
blockspergrid_y = math.ceil(height / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)

redraw = False
x = np.linspace(center[0]-xrange/2, center[0]+xrange/2, width, dtype=np.float32)
y = np.linspace(center[1]-yrange/2, center[1]+yrange/2, height, dtype=np.float32)
pixels = np.zeros((width, height), dtype=np.float32)

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
    global c, xrange, yrange, center, redraw
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)

    if key == glfw.KEY_A:
        redraw = True
        c = np.complex64(cmath.rect(.7885, cmath.phase(c)-0.005))
    if key == glfw.KEY_D:
        redraw = True
        c = np.complex64(cmath.rect(.7885, cmath.phase(c)+0.005))

    if key == glfw.KEY_W:
        redraw = True
        xrange /= 1.1
        yrange = xrange * ratio
        print(xrange)
    if key == glfw.KEY_S:
        redraw = True
        xrange *= 1.1
        yrange = xrange * ratio
        print(xrange)

    if key == glfw.KEY_UP:
        redraw = True
        center[0] += xrange*0.01
        print(center)
    if key == glfw.KEY_DOWN:
        redraw = True
        center[0] -= xrange*0.01
        print(center)
    if key == glfw.KEY_LEFT:
        redraw = True
        center[1] -= yrange*0.01
        print(center)
    if key == glfw.KEY_RIGHT:
        redraw = True
        center[1] += yrange*0.01
        print(center)

def cursor_position_fallback(window, xpos, ypos):
    global width, height
    print(xpos-width/2, ypos-height/2)

@cuda.jit
def kernel(A, x, y, c):
    i, k = cuda.grid(2)
    x_size = A.shape[0]
    y_size = A.shape[1]

    threshold = int32(100)

    if i < x_size and k < y_size:
        count = int32(0)
        dist = float32(0)
        z = complex64(x[i] + y[k]*1j)
        while dist < 2 and count < threshold:
            z = z**2+c
            count += 1
            dist = abs(z)
        A[i,k] = count / threshold

def color_mapper(A):
    return colormap(A)[...,:3]

def calculatePixels():
    global x, y, c, blockspergrid, threadsperblock, pixels

    gpu_pixels = cuda.to_device(pixels)
    kernel[blockspergrid, threadsperblock](
        gpu_pixels,
        cuda.to_device(x), cuda.to_device(y), 
        c)
    pixels = gpu_pixels.copy_to_host()
    

glfw.set_key_callback(window, key_callback)
glfw.set_cursor_pos_callback(window, cursor_position_fallback)

while not glfw.window_should_close(window):
    # Render here, e.g. using pyOpenGL
    glClear(GL_COLOR_BUFFER_BIT)
    
    
    # listens to inputs
    glfw.poll_events()

    if redraw:
        x = np.linspace(center[0]-xrange/2, center[0]+xrange/2, width, dtype=np.float32)
        y = np.linspace(center[1]-yrange/2, center[1]+yrange/2, height, dtype=np.float32)
        calculatePixels()
        glDrawPixels(width, height, GL_RED, GL_FLOAT, pixels)
        redraw=False
    else:
        pixels = (pixels - pixels.min())/pixels.max()
        glDrawPixels(width, height, GL_RGB, GL_FLOAT, colormap(pixels)[...,:3])

    # exchanges back and front buffer
    glfw.swap_buffers(window)

glfw.terminate()