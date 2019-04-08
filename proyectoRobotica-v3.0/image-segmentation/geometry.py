# coding=UTF-8

def dist(p,q):
    return (p[0]-q[0])**2 + (p[1]-q[1])**2

# Área signada del triángulo definido por los puntos a, b y c.
def sarea(a,b,c):
    return 0.5*((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))