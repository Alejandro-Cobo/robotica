import numpy as np
import scipy.spatial.distance as dst

class classifMahalanobis:
    def __init__(self):
        pass
    
    def fit(self,x,y):
        self.labels = np.unique(y)
        self.means = np.zeros((len(self.labels), x.shape[1]))
        self.invcov = np.empty((x.shape[1],x.shape[1],len(self.labels)))
        for i in xrange(len(self.labels)):
            self.means[i] = np.mean(x[y == self.labels[i]], axis=0)
            self.invcov[:,:,i] = np.linalg.pinv( np.cov(x[y==self.labels[i]],rowvar=False) )
        return self
    
    def predict(self,x):
        distances = np.empty((x.shape[0],self.labels.shape[0]))
        for i in xrange(x.shape[0]):
            for j in xrange(len(self.labels)):
                distances[i,j] = dst.mahalanobis(x[i],self.means[j],self.invcov[:,:,j])
        return self.labels[np.argmin(distances, axis=1)]