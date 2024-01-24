import pyglet.window.key
from pyglet.gl import *
import random
import cv2
import numpy as np
import math
from itertools import filterfalse

batch = pyglet.graphics.Batch()

triCol = [0.0, 0.0, 0.0]
glColor3f(triCol[0], triCol[1], triCol[2])
glClearColor(252, 13, 200, 0)
glClear(GL_COLOR_BUFFER_BIT)
window = pyglet.window.Window(750, 750)

x = 0
y = 0
z = -5
angle = 0

startX = 1
startY = 1

particles = []

board = True


# provjera boje pozadine
def is_white_background(image):
    # pretvorba u grayscale

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # prag za bijelu boju
    threshold = 240

    whites = cv2.countNonZero(cv2.inRange(gray, threshold, 255))

    # vidi koliko je bijelih u odnosu na sve piksele
    total = gray.shape[0] * gray.shape[1]
    white_ratio = whites / total

    # ako je vise od pola bijelih, bijela je pozadina
    threshold = 0.5

    if white_ratio > threshold:
        return True

    return False


src = cv2.imread('explosion.bmp', 1)

white = is_white_background(src)

tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

result = None

if white:

    # odredi prag za bilu boju
    _, alpha = cv2.threshold(tmp, 240, 255, cv2.THRESH_BINARY)
    alpha = cv2.bitwise_not(alpha)
    b, g, r = cv2.split(src)
    rgba = [b, g, r, alpha]
    result = cv2.merge(rgba, 4)
else:

    # prag za crnu boju
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b, g, r, alpha]
    result = cv2.merge(rgba, 4)

cv2.imwrite("removed.png", result)

slika = pyglet.image.load('removed.png')


def rotacija(pocetak, cilj):
    os = np.array([[pocetak[1] * cilj[2] - (cilj[1] * pocetak[2])], [-(pocetak[0] * cilj[2] - (cilj[0] * pocetak[2]))],
                   [pocetak[0] * cilj[1] - (cilj[0] * pocetak[1])]])

    cosfi = (np.dot(pocetak, cilj)) / ((math.sqrt(pocetak[0] ** 2 + pocetak[1] ** 2 + pocetak[2] ** 2)) * (
        math.sqrt(cilj[0] ** 2 + cilj[1] ** 2 + cilj[2] ** 2)))
    return (os, math.degrees(math.acos(cosfi)))


class Particle:

    def __init__(self):
        self.text = slika.get_texture()

        self.picture = self.text
        self.x = random.uniform(-20, 20)
        self.y = random.uniform(-20, 20)
        self.z = random.uniform(-10, 10)
        self.w = 5
        self.h = 7

        # random promjene u x i y smjeru
        self.age = 0
        self.finalAge = random.uniform(50,100)

    def increase_age(self, dt):
        global particles

        self.age += 1
        if self.age > self.finalAge*(2/3):
            self.w *= 0.8
            self.h *= 0.8

        self.x += random.uniform(-1, 1)
        self.y += random.uniform(-1, 1)
        self.z += random.uniform(-1, 1)

    def draw(self):

        glEnable(pyglet.gl.GL_BLEND)
        glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)

        glPushMatrix()
        glTranslatef(self.x, self.y, self.z)

        if board:
            os, kut = rotacija([self.x - x, self.y - y, self.z - z], [0, 0, -1])
            glRotatef(kut, os[0], os[1], os[2])

        self.picture.blit(self.x, self.y, self.z, width=self.w, height=self.h)

        glDisable(GL_BLEND)

        glTranslatef(-self.x, -self.y, -self.z)

        glPopMatrix()


def tooOld(x: Particle):
    return x.age > x.finalAge


def increase_age(dt):
    global particles

    # umiranje prestarih cestica
    particles[:] = filterfalse(tooOld, particles)

    particles.append(Particle())

    for particle in particles:
        particle.increase_age(dt)


@window.event
def on_key_press(symbol, modifiers):
    global x, y, z, angle

    if symbol == pyglet.window.key.RIGHT:
        x += 5
    elif symbol == pyglet.window.key.LEFT:
        x -= 5
    elif symbol == pyglet.window.key.UP:
        y += 5
    elif symbol == pyglet.window.key.DOWN:
        y -= 5
    elif symbol == pyglet.window.key.O:
        z += 5
    elif symbol == pyglet.window.key.P:
        z -= 5
    elif symbol == pyglet.window.key.L:
        angle += 5
    elif symbol == pyglet.window.key.R:
        angle -= 5


@window.event
def on_mouse_press(x, y, button, modifiers):
    global startX, startY

    startX = (x - window.width / 2) / 10
    startY = (y - window.height / 2) / 10


@window.event
def on_draw():
    global x, y, z, angle, particles

    glClear(GL_COLOR_BUFFER_BIT)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(90, 1, 0.2, 100)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(x, y, z)
    glRotatef(angle, 0, 1, 0)

    for particle in particles:
        particle.draw()


@window.event
def on_resize(width, height):
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, width, 0, height, - 1, 1)
    glMatrixMode(GL_MODELVIEW)


pyglet.clock.schedule_interval(increase_age, 0.3)
pyglet.app.run()
