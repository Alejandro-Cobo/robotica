# coding=UTF-8

def dist(p,q):
    """
    Devuelve el cuadrado de la distancia euclídea entre los puntos p y q.
    """
    return (p[0]-q[0])**2 + (p[1]-q[1])**2

def sarea(a,b,c):
    """
    Devuelve el doble del área signada del triángulo definido por los puntos a, b y c.
    """
    return (b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])
    