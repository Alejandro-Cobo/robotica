import numpy as np
import scipy.spatial.distance as dst

class classifMahalanobis:
    """
    Clase que implementa el clasificador de la distancia de Mahalanobis para el
    reconocimiento de iconos.
    """
    def __init__(self):
        pass
    
    def fit(self,x,y):
        """
        Entrena el clasificador.

        Argumentos:
            x (numpy.ndarray): array con las instancias de los datos sin la variable clase.
            y (numpy.ndarray): array con las clases de los datos en x.

        Devuelve:
            classifMahalanobis: instancia del clasificador entrenado.
        """
        self.labels = np.unique(y)
        self.means = np.zeros((len(self.labels), x.shape[1]))
        self.invcov = np.empty((x.shape[1],x.shape[1],len(self.labels)))
        for i in range(len(self.labels)):
            self.means[i] = np.mean(x[y == self.labels[i]], axis=0)
            self.invcov[:,:,i] = np.linalg.inv( np.cov(x[y==self.labels[i]],rowvar=False) )
        return self
    
    def predict(self,x):
        """
        Estima las clases de los datos pasados como argumento.

        Argumentos:
            x (numpy.ndarray): matriz con los datos a predecir. Cada fila representa una
                instancia, y cada columna una variable.
    
        Devuelve:
            numpy.ndarray: array con las clases predichas para cada dato.
        """
        distances = np.empty((x.shape[0],self.labels.shape[0]))
        for i in range(x.shape[0]):
            for j in range(len(self.labels)):
                distances[i,j] = dst.mahalanobis(x[i],self.means[j],self.invcov[:,:,j])
        return int(self.labels[np.argmin(distances, axis=1)])
        