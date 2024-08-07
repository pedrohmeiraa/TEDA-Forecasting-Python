from padasip.filters.base_filter import AdaptiveFilter

import pandas as pd
import numpy as np
import padasip as pa

np.random.seed(0)

class DataCloud:
  N=0
  def __init__(self, nf, mu, w, x):
      self.n=1
      self.nf=nf
      self.mean=x
      self.mu = mu
      self.w = w
      self.variance=0
      self.pertinency=1
      self.faux = pa.filters.FilterRLS(n=self.nf, mu=self.mu, w=self.w)
      DataCloud.N+=1
  def addDataCloud(self,x):
      self.n=2
      self.mean=(self.mean+x)/2
      self.variance=((np.linalg.norm(self.mean-x))**2)
  def updateDataCloud(self,n,mean,variance, faux):
      self.n=n
      self.mean=mean
      self.variance=variance
      self.faux = faux
  
 
class TEDAForecasting:
  c= np.array([DataCloud(nf=2, mu=0.9, w=[0,0], x=0)],dtype=DataCloud)
  alfa= np.array([0.0],dtype=float)
  intersection = np.zeros((1,1),dtype=int)
  listIntersection = np.zeros((1),dtype=int)
  matrixIntersection = np.zeros((1,1),dtype=int)
  relevanceList = np.zeros((1),dtype=int)
  k=1

  def __init__(self, window, mu, wI, threshold):
    TEDAForecasting.window = window
    TEDAForecasting.mu = mu
    TEDAForecasting.wI = wI
    TEDAForecasting.threshold = threshold


    TEDAForecasting.alfa= np.array([0.0],dtype=float)
    TEDAForecasting.intersection = np.zeros((1,1),dtype=int)
    TEDAForecasting.listIntersection = np.zeros((1),dtype=int)
    TEDAForecasting.relevanceList = np.zeros((1),dtype=int)
    TEDAForecasting.matrixIntersection = np.zeros((1,1),dtype=int)
    TEDAForecasting.k=1
    TEDAForecasting.classIndex = [[1.0],[1.0]] #<========== try in another moment: [np.array(1.0),np.array(1.0)]
    TEDAForecasting.argMax = []
    TEDAForecasting.RLSF_Index = []
    TEDAForecasting.Ypred = []
    TEDAForecasting.Ypred_STACK = []
    TEDAForecasting.Ypred_POND = []
    TEDAForecasting.Ypred_MAJOR = []
    TEDAForecasting.X_ant = np.zeros((1, TEDAForecasting.window), dtype=float)
    TEDAForecasting.NumberOfFilters = []
    TEDAForecasting.NumberOfDataClouds = []
    
    np.random.seed(0)
    TEDAForecasting.random_factor = np.random.rand(TEDAForecasting.window-1, TEDAForecasting.window)
    #TEDAForecasting.random_factor = np.array([[0.00504779, 0.99709118]])

    if (TEDAForecasting.wI == "relu"): #He
      factor = np.sqrt(2/(TEDAForecasting.window-1))
    elif (TEDAForecasting.wI == "tanh1"): #Xavier
      factor = np.sqrt(1/(TEDAForecasting.window-1))
    elif (TEDAForecasting.wI == "tanh2"): #Yoshua
      factor = np.sqrt(2/((2*TEDAForecasting.window)-1))
    elif (TEDAForecasting.wI == "zero"):
      factor = 0
    else: #Utiliza a Formula do "He" como DEFAULT
      factor = np.sqrt(2/TEDAForecasting.window-1)
           
    TEDAForecasting.w_init = TEDAForecasting.random_factor*factor 
    TEDAForecasting.w_init = TEDAForecasting.w_init[0].tolist()
    #print("w_init do TEDA: ", TEDAForecasting.w_init)
    TEDAForecasting.c = np.array([DataCloud(nf=TEDAForecasting.window, mu=TEDAForecasting.mu, w=TEDAForecasting.w_init, x=0)],dtype=DataCloud)
    TEDAForecasting.f0 = pa.filters.FilterRLS(TEDAForecasting.window, mu=TEDAForecasting.mu, w=TEDAForecasting.w_init)


  def mergeClouds(self):
    i=0
    while(i<len(TEDAForecasting.listIntersection)-1):
      #print("i do merge",i)
      #print(TEDAForecasting.listIntersection)
      #print(TEDAForecasting.matrixIntersection)
      merge = False
      j=i+1
      while(j<len(TEDAForecasting.listIntersection)):
        #print("j do merge",j)
        #print("i",i,"j",j,"l",np.size(TEDAForecasting.listIntersection),"window",np.size(TEDAForecasting.matrixIntersection),"c",np.size(TEDAForecasting.c))
        if(TEDAForecasting.listIntersection[i] == 1 and TEDAForecasting.listIntersection[j] == 1):
          TEDAForecasting.matrixIntersection[i,j] = TEDAForecasting.matrixIntersection[i,j] + 1;
          #print(TEDAForecasting.matrixIntersection)
        nI = TEDAForecasting.c[i].n
        nJ = TEDAForecasting.c[j].n
        #print("I: ",list(TEDAForecasting.c).index(TEDAForecasting.c[i]), ". J: ",list(TEDAForecasting.c).index(TEDAForecasting.c[j]))
        meanI = TEDAForecasting.c[i].mean
        meanJ = TEDAForecasting.c[j].mean
        varianceI = TEDAForecasting.c[i].variance
        varianceJ = TEDAForecasting.c[j].variance
        nIntersc = TEDAForecasting.matrixIntersection[i,j]
        fauxI = TEDAForecasting.c[i].faux    #fauxI = TEDAForecasting.RLS_Filters[i]
        fauxJ = TEDAForecasting.c[j].faux    #fauxJ = TEDAForecasting.RLS_Filters[j]
        
        wI = fauxI.getW()
        wJ = fauxJ.getW()
        #print("wJ: ", wJ)
        
        dwI = fauxI.getdW()
        dwJ = fauxJ.getdW()
                
        W = (nI*wI)/(nI + nJ) + (nJ*wJ)/(nI + nJ)

        if (nIntersc > (nI - nIntersc) or nIntersc > (nJ - nIntersc)):
          #print(nIntersc, "(nIntersc) >", nI, "-", nIntersc, "=", nI - nIntersc, "(nI - nIntersc) OR", nIntersc, "(nIntersc) >", nJ, "-", nIntersc, "=", nJ - nIntersc, "(nJ - nIntersc)")
          
          #print("(nIntersc) =",nIntersc, ",nI =", nI, ",nJ =", nJ)
          #print("Juntou!")

          merge = True
          #update values
          n = nI + nJ - nIntersc
          mean = ((nI * meanI) + (nJ * meanJ))/(nI + nJ)
          variance = ((nI - 1) * varianceI + (nJ - 1) * varianceJ)/(nI + nJ - 2)
          #faux = fauxI #Considerando o de maior experiencia (mais amostras)
          #faux = pa.filters.FilterRLS(2, mu=mu_, w = W) #Considerando a inicialização dos pesos (W)
                       
          if (nI >= nJ):
            fauxI.mergeWith(fauxJ, nI, nJ)
            faux = fauxI

          else:
            fauxJ.mergeWith(fauxI, nI, nJ)
            faux = fauxJ
          
          #TEDAForecasting.RLS_Filters.pop()

          newCloud = DataCloud(nf=TEDAForecasting.window, mu=TEDAForecasting.mu, w=TEDAForecasting.w_init, x=mean)
          newCloud.updateDataCloud(n,mean,variance, faux)
          
          #atualizando lista de interseção
          TEDAForecasting.listIntersection = np.concatenate((TEDAForecasting.listIntersection[0 : i], np.array([1]), TEDAForecasting.listIntersection[i + 1 : j],TEDAForecasting.listIntersection[j + 1 : np.size(TEDAForecasting.listIntersection)]),axis=None)
          #print("listInters dps de att:", TEDAForecasting.listIntersection)
          #atualizando lista de data clouds
          #print("dentro do if do merge antes", TEDAForecasting.c)
          TEDAForecasting.c = np.concatenate((TEDAForecasting.c[0 : i ], np.array([newCloud]), TEDAForecasting.c[i + 1 : j],TEDAForecasting.c[j + 1 : np.size(TEDAForecasting.c)]),axis=None)
          #print("dentro do if do merge dps do concate", TEDAForecasting.c)

          #update  intersection matrix
          M0 = TEDAForecasting.matrixIntersection
          #Remover linhas 
          M1=np.concatenate((M0[0 : i , :],np.zeros((1,len(M0))),M0[i + 1 : j, :],M0[j + 1 : len(M0), :]))
          #remover colunas
          M1=np.concatenate((M1[:, 0 : i ],np.zeros((len(M1),1)),M1[:, i+1 : j],M1[:, j+1 : len(M0)]),axis=1)
          #calculando nova coluna
          col = (M0[:, i] + M0[:, j])*(M0[: , i]*M0[:, j] != 0)
          col = np.concatenate((col[0 : j], col[j + 1 : np.size(col)]))
          #calculando nova linha
          lin = (M0[i, :]+M0[j, :])*(M0[i, :]*M0[j, :] != 0)
          lin = np.concatenate((lin[ 0 : j], lin[j + 1 : np.size(lin)]))
          #atualizando coluna
          M1[:,i]=col
          #atualizando linha
          M1[i,:]=lin
          M1[i, i + 1 : j] = M0[i, i + 1 : j] + M0[i + 1 : j, j].T;   
          TEDAForecasting.matrixIntersection = M1
          #print(TEDAForecasting.matrixIntersection)
        j += 1
      if (merge):
        i = 0
      else:
        i += 1
				
  def run(self,X):
    TEDAForecasting.listIntersection = np.zeros((np.size(TEDAForecasting.c)),dtype=int)
    #print("k=", TEDAForecasting.k)
    if TEDAForecasting.k==1:
      TEDAForecasting.c[0]=DataCloud(nf=TEDAForecasting.window, mu=TEDAForecasting.mu, w=TEDAForecasting.w_init, x=X)
      TEDAForecasting.argMax.append(0)
      TEDAForecasting.c[0].faux = TEDAForecasting.f0 #TEDAForecasting.RLS_Filters = [TEDAForecasting.f0] 
      TEDAForecasting.RLSF_Index.append(0)
      TEDAForecasting.X_ant = X
      #TEDAForecasting.c[0].faux.adapt(X[1], TEDAForecasting.X_ant)

    elif TEDAForecasting.k==2:
      TEDAForecasting.c[0].addDataCloud(X)
      TEDAForecasting.argMax.append(0)
      TEDAForecasting.RLSF_Index.append(0)
      TEDAForecasting.X_ant = X
      #TEDAForecasting.c[0].faux.adapt(X[1], TEDAForecasting.X_ant)
    
    elif TEDAForecasting.k>=3:
      i=0
      createCloud = True
      TEDAForecasting.alfa = np.zeros((np.size(TEDAForecasting.c)),dtype=float)

      for data in TEDAForecasting.c:
        n= data.n + 1
        mean = ((n-1)/n)*data.mean + (1/n)*X
        variance = ((n-1)/n)*data.variance +(1/n)*((np.linalg.norm(X-mean))**2)
        eccentricity=(1/n)+((mean-X).T.dot(mean-X))/(n*variance)
        typicality = 1 - eccentricity
        norm_eccentricity = eccentricity/2
        norm_typicality = typicality/(TEDAForecasting.k-2)
        faux_ = data.faux
        #faux_.adapt(X[-1], TEDAForecasting.X_ant)
        
        if (norm_eccentricity<=(TEDAForecasting.threshold**2 +1)/(2*n)): #Se couber dentro da Cloud
          
          data.updateDataCloud(n,mean,variance, faux_)
          TEDAForecasting.alfa[i] = norm_typicality
          createCloud= False
          TEDAForecasting.listIntersection.itemset(i,1)
          #print("pesos=", data.faux.w)
          #print("x_ant=", TEDAForecasting.X_ant)
          #print("dentro da cloud")
          faux_.adapt(X[-1], TEDAForecasting.X_ant)
        #TEDAForecasting.c[i].faux.adapt(X[-1], TEDAForecasting.X_ant) #data.faux.adapt(X[-1], TEDAForecasting.X_ant) #data.faux.adapt(X[-1], TEDAForecasting.X_ant)
              
        else: #Se nao couber
          TEDAForecasting.alfa[i] = norm_typicality
          TEDAForecasting.listIntersection.itemset(i,0)
          #print("fora da cloud")
          #print("fora -> i:", i, " - Filtro: ", faux_)
          #faux_.adapt(X[-1], TEDAForecasting.X_ant)
          #TEDAForecasting.c[i].faux.adapt(X[-1], TEDAForecasting.X_ant) #data.faux.adapt(X[-1], TEDAForecasting.X_ant)
        i+=1

      if (createCloud):
        #print("no if de criar TEDAForecasting:", TEDAForecasting.c)
        TEDAForecasting.c = np.append(TEDAForecasting.c,DataCloud(nf=TEDAForecasting.window, mu=TEDAForecasting.mu, w=TEDAForecasting.w_init, x=X))
        #print("dps do if TEDAForecasting:", TEDAForecasting.c)
        TEDAForecasting.listIntersection = np.insert(TEDAForecasting.listIntersection,i,1)
        TEDAForecasting.matrixIntersection = np.pad(TEDAForecasting.matrixIntersection, ((0,1),(0,1)), 'constant', constant_values=(0))
        #print("DataCloud Created!")
        #TEDAForecasting.RLS_Filters.append(pa.filters.FilterRLS(TEDAForecasting.window, mu=TEDAForecasting.mu, w=TEDAForecasting.w_init))


      #print("TEDAForecasting antes do Merge:", TEDAForecasting.c)
      
      #print("TEDAForecasting dps do Merge:", TEDAForecasting.c)
      TEDAForecasting.NumberOfFilters.append(len(TEDAForecasting.c))
      TEDAForecasting.relevanceList = TEDAForecasting.alfa /np.sum(TEDAForecasting.alfa)
      TEDAForecasting.argMax.append(np.argmax(TEDAForecasting.relevanceList))
      TEDAForecasting.classIndex.append(TEDAForecasting.alfa)
      #print("Alfa", TEDAForecasting.alfa)
      #print("argmax", np.argmax(TEDAForecasting.relevanceList))
      #print("relevance list: ", TEDAForecasting.relevanceList)
          
      filtro_usado = TEDAForecasting.c[np.argmax(TEDAForecasting.relevanceList)].faux

      #print("filtro_usado:", filtro_usado)
      #print("_______")
      
      self.mergeClouds()
    
    
      F_used = []
      N_used = []
      YX = []
      NX = []
    
      c=0

      for x in TEDAForecasting.c:
        c = c+1
        fx = x.faux
        Nx = x.n
  
        F_used.append(fx)
        N_used.append(Nx)
      
      for i in range(0, len(F_used)): 
        
        yx_pred = F_used[i].predict(X)
        nx_pred = N_used[i]
        YX.append(yx_pred)
        NX.append(nx_pred)
        
      #Stacking
      y_pred_stack = sum(YX)/len(YX)
      #print("ypred_bag: ", y_pred_bag)
        
      #Pondering     
      y_pred_pond = np.sum(np.multiply(YX, NX))/np.sum(NX)
      #print("ypred_pond", y_pred_pond)
      
      #Best of all
      y_pred_major = filtro_usado.predict(X)
      #print("ypred_major", y_pred_major)        
      #print("___________")

      TEDAForecasting.Ypred.append(y_pred_pond)
      TEDAForecasting.Ypred_STACK.append(y_pred_stack)
      TEDAForecasting.Ypred_POND.append(y_pred_pond)
      TEDAForecasting.Ypred_MAJOR.append(y_pred_major)
        
      TEDAForecasting.RLSF_Index.append(np.argmax(TEDAForecasting.relevanceList))
      TEDAForecasting.X_ant = X
      
    TEDAForecasting.k=TEDAForecasting.k+1