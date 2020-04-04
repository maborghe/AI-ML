from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def clear_labels(ax, x, y):
  for i, c in enumerate(x):
    ax[c].set_xticklabels([])
    ax[c].set_xlabel('')    
  for i, c, in enumerate(y):
    ax[c].set_yticklabels([])
    ax[c].set_ylabel('')         
    
def truncate(f):
    '''Truncates/pads a float f to 2 decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, 2)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*2)[:2]])

def denormalize_labels(ax, p_mean, p_variance):
    xloc, xlabels = plt.xticks()
    dloc = xloc*p_variance[0] + p_mean[0]
    for ii, f in enumerate(dloc):
        dloc[ii] = truncate(dloc[ii])        
    ax.set_xticklabels(dloc)
    
    yloc, ylabels = plt.yticks()
    dloc = yloc*p_variance[1] + p_mean[1]
    for ii, f in enumerate(dloc):
        dloc[ii] = truncate(dloc[ii])        
    ax.set_yticklabels(dloc)    

X, y = load_wine(True)
x = X[:,:2]
xNotTest, xTest, yNotTest, yTest = train_test_split(x, y, test_size=0.3)
xTrain, xVal, yTrain, yVal = train_test_split(xNotTest, yNotTest, test_size=0.28)
XTrain, XTest, YTest, YTest = train_test_split(X, y, test_size=0.3)

# Normalize data
scaler = StandardScaler()
scaler.fit(xNotTest)
xTrain = scaler.transform(xTrain)
xVal = scaler.transform(xVal)
xTest = scaler.transform(xTest)
xNotTest = scaler.transform(xNotTest)
mean = scaler.mean_
variance = scaler.var_


# Setup plot
x_min, x_max = xTrain[:, 0].min() - 1, xTrain[:, 0].max() + 1
y_min, y_max = xTrain[:, 1].min() - 1, xTrain[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                         np.arange(y_min, y_max, .02))
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

KNN = 1
SVM = 1   
_PCA = 1         
             
bestK = 0
bestC = [0, 0] # linear, rbf
               
# 1. KNN
if KNN:
    K = [1,3,5,7]
    Kscore = [None]*len(K)
    plt.figure()
    ax = [None]*len(K)
    for i, k in enumerate(K):    
        clf = KNeighborsClassifier(n_neighbors=k).fit(xTrain, yTrain)
        Kscore[i] = clf.score(xVal, yVal)
        
        # Create plot        
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)        
            
        ax[i] = plt.subplot(2, 2, i + 1)    
        plt.xlabel('Alcohol')
        plt.ylabel('Malic acid')
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
        plt.scatter(xTrain[:, 0], xTrain[:, 1], c=yTrain, cmap=cmap_bold,
                    edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        denormalize_labels(ax[i], mean, variance)      
    
    clear_labels(ax, [0, 1], [1, 3])
    plt.savefig('knnPredPlot', dpi=250)
    
    # plot Kscores
    plt.figure()
    plt.scatter(K, Kscore)
    plt.xlabel('K')
    plt.ylabel('accuracy')
    plt.savefig('knnScorePlot', dpi=250)
    
    # evaluate the best K on the test set
    bestKIndex = np.asarray(Kscore).argmax()
    bestK = K[bestKIndex]
    clf = KNeighborsClassifier(n_neighbors=bestK).fit(xTrain, yTrain)
    KNNtestScore = clf.score(xTest, yTest)
    print("KNN test score: %f" % (KNNtestScore))

# 2. and 3. linear and rbf SVM
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
SVMscore = [None]*len(C)        
if SVM:        
    for c, ker in enumerate(["linear", "rbf"]):               
        ax = [None]*len(C)                
        plt.figure()
        for i, cc in enumerate(C):
            clf = SVC(C=cc, gamma='scale', kernel=ker).fit(xTrain, yTrain)
            SVMscore[i] = clf.score(xVal, yVal)
            ax[i] = plt.subplot(3, 3, i+1) 
            plt.xlabel('Alcohol')
            plt.ylabel('Malic acid')
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap=cmap_light)
            plt.scatter(xTrain[:, 0], xTrain[:, 1], c=yTrain, cmap=cmap_bold,
                        edgecolor='k', s=20)   
            
            denormalize_labels(ax[i], mean, variance) 
                        
        clear_labels(ax, [0, 1, 2, 3], [1, 2, 4, 5])
        plt.savefig(ker+'PredPlot', dpi=250)
        
        # plot Kscores
        plt.figure()
        plt.xlim(0.0001, 10000)
        plt.xscale("log")      
        plt.xlabel('C')
        plt.ylabel('accuracy')
        plt.scatter(C, SVMscore)
        plt.savefig(ker+'ScorePlot', dpi=250)
        
        # evaluate the best C on the test set
        bestCIndex = np.asarray(SVMscore).argmax()
        bestC[c] = C[bestCIndex]
        clf = SVC(C=bestC[c], gamma='scale', kernel=ker).fit(xTrain, yTrain)
        SVMtestScore = clf.score(xTest, yTest)
        print(ker + " test score: %f" % (SVMtestScore))
        print()
                           
            
    #4. Grid search
    GridScore = [SVMscore, [None]*len(C)]
    for i, cc in enumerate(C):
        clf = SVC(C=cc, gamma='auto', kernel='rbf').fit(xTrain, yTrain)
        GridScore[1][i] = clf.score(xVal, yVal)
    
    bestVal = np.asmatrix(GridScore).argmax()
    bestCGrid = C[bestVal % len(C)]
    bestGamma = 'scale' if bestVal//len(C) == 0 else 'auto'
    clf = SVC(C=bestCGrid, gamma=bestGamma, kernel='rbf').fit(xTrain, yTrain)
    plt.figure()
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.scatter(xTrain[:, 0], xTrain[:, 1], c=yTrain, cmap=cmap_bold,
                edgecolor='k', s=20)      

    denormalize_labels(plt.axes(), mean, variance)
    plt.xlabel('Alcohol')
    plt.ylabel('Malic acid')    
    plt.savefig('gridPredPlot', dpi=250)
    gridTestScore = clf.score(xTest, yTest)
    print("best params: C=%f, gamma=%s" %(bestCGrid, bestGamma))
    print("grid test score: %f" % (gridTestScore))
    print()    


    # 5. Grid search with 5-Fold CV
    # Merge train and validation sets         
    xTrain = xNotTest
    yTrain = yNotTest
    parameters = {'gamma':('auto', 'scale'), 'C':C}
    clf = GridSearchCV(SVC(kernel="rbf"), parameters, cv=5, iid='false').fit(xTrain, yTrain)
    plt.figure()
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.scatter(xTrain[:, 0], xTrain[:, 1], c=yTrain, cmap=cmap_bold,
                edgecolor='k', s=20)    
    denormalize_labels(plt.axes(), mean, variance)
    plt.xlabel('Alcohol')
    plt.ylabel('Malic acid')    
    plt.savefig('cvGridPredPlot', dpi=250)
    gridTestScore = clf.score(xTest, yTest)
    bestParams = ', '.join("{!s}={!r}".format(key,val) for (key,val) in clf.best_params_.items())    
    print("CV best params: " + bestParams)
    print("CV grid test score: %f" % (gridTestScore))
    
# 6. PCA    
if _PCA:
    pca = PCA(n_components=2)
    pca.fit(xNotTest)
    pcaTrain = pca.transform(xTrain)
    pcaVal = pca.transform(xVal)
    pcaTest = pca.transform(xTest)
    pcaNotTest = pca.transform(xNotTest)
    
    clf = KNeighborsClassifier(n_neighbors=bestK).fit(pcaTrain, yTrain)      
    score = clf.score(pcaTest, yTest)
    print("PCA-KNN score: %f" %(score))
    
    clf = SVC(C=bestC[0], gamma='scale', kernel='linear').fit(pcaTrain, yTrain)
    score = clf.score(pcaTest, yTest)
    print("PCA-linear SVM score: %f" %(score))
    
    clf = SVC(C=bestC[1], gamma='scale', kernel='rbf').fit(pcaTrain, yTrain)
    score = clf.score(pcaTest, yTest)
    print("PCA-RBF SVM score: %f" %(score)) 
