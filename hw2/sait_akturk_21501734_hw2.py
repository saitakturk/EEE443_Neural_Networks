import numpy as np 
import matplotlib.pyplot as plt
import scipy.io
from scipy import optimize
from sklearn.metrics import confusion_matrix
import itertools
import string
import sys

#load datas
data  = scipy.io.loadmat('assign1_data1.mat')
data2 = scipy.io.loadmat('assign1_data2.mat')

def img_data_generator(datax):
    """
    This function reads images data which are numpy.darray format 
    and flatten each image to make 1D vector from grayscale image
    and lastly normalize each row vector from  [1,255] --> [0,1]

    Args:
        data: images of data

    Returns:
        numpy.darray that contains image with preprocessed for simple nn applications
    """
    X = []
    length_of_data = len(datax[0,0,:])
  
    for i in range(length_of_data):
        #read the data 
        img = datax[:,:,i].flatten()
        #add one more dimension for convention 
        #img = np.expand_dims(img, axis = 1 )
        # make row vector each image with transposing column vector
        img = img.T
       
        #normalize the features [0, 255] ---> [0 ,1]
        img = img.astype(np.float64)/ 255.0
        X.append(img)
    
    X = np.array(X)
    return X


def labels_generator(labels, polarity = 'unipolar'):
    """
    This function make "0" labels to "-1" since we are using tanh(x) and transpose y labels for simple nn
    
    Args:
        labels:  labels data
        polarity : change some properties according to activation fucntion that used 

    Returns:
        transpose of labels to be convenient implementation of nn
    """
    #convert np.uint8 to np.float64 to make '0' to '-1'
    if( polarity == 'bipolar'):
        labels_temp = labels.astype(np.float64)
        #make "0" to '-1'
        labels_temp[ labels_temp == 0 ] = -1 

        if(labels_temp.shape[0] != 1):
            return labels_temp
        else:
            return labels_temp.T

    else : 
        if(labels.shape[0] != 1):
            return labels
        else:
            return labels.T

    
def random_shuffle(X, y):
    """
    This function make random shuffle for two data exact same shuffle.
    Also, As a remainder, this function create random shuffled data
    each time different 
    
    Args:
        X:  data
        y: labels

    Returns:
        shuffled datas and labels
    """
    #random shuffle data 
    idx = np.random.permutation(len(y))
    return  X[idx], y[idx] 

def normal_weight_and_bias_initializer( w_size = (1024,1), b_size = (1,1), loc = 0.0, scale = 0.01):
    """
    This function initialize the weights and the biases with Gaussian distribution
    
    Args:
        w_size:  weight size
        b_size:  bias size 
        loc : mean 
        scale : std 
    Returns:
        weights and biases
    """
    np.random.seed(0)
    #initialize weights
    W = np.random.normal(size = w_size, loc = loc, scale = scale ).astype(np.float64)
    #initialize biases
    b = np.random.normal(size = b_size, loc = loc , scale = scale ).astype(np.float64)
    return W, b  
def glorot_weight_and_bias_initializer(w_size, b_size ):
    """
    This function initialize the weights and the biases with xavier initialization
    
    Args:
        w_size:  weight size
        b_size:  bias size 
    Returns:
        weights and biases
    """
    np.random.seed(42)
    #initialize weights
    range_w0 = np.sqrt(6/ (w_size[0] + w_size[1]))
    W = np.random.uniform( size=(w_size[0], w_size[1]), low = -range_w0, high=range_w0).astype(np.float64)
    #initialize biases
    b = np.random.uniform( size=(b_size[0],b_size[1]), low = -range_w0, high=range_w0).astype(np.float64)
    return { 'w' :W, 'b' : b }  
def xavier_weight_and_bias_initializer( w_size, b_size): 
    """
    This function initialize the weights and the biases with xavier initialization
    
    Args:
        w_size:  weight size
        b_size:  bias size 
    Returns:
        weights and biases
    """
    #initialize weights
    W = (np.random.randn( w_size[0], w_size[1]) * np.sqrt(1/ (w_size[0]))).astype(np.float64)
    #initialize biasesm
    b = (np.random.randn(b_size[0],b_size[1]) * np.sqrt(1/ (b_size[1]))).astype(np.float64)
    return { 'w' :W, 'b' : b }  

def tanh_activation(z, derive = False ):
    """
    This function calculate tanh function and its derivative
    !IMPORTANT TO READ! Giving x for derivative of tanh(x) is calculated before therefore no need to recalculate
    Args:
        z:  matrix that will calculate
        derive: check user wants to derive of tanh(x) function
    Returns:
        return output of tanh(x) function of its derivative
    """
    if(derive == True ):
        #derive of tanh(x)
        return 1 - ( z ** 2 )
    #tanh(x)
    return ( np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def sigmoid_activation(z, derive = False ):
    """
    This function calculate sigmoid function and its derivative
    !IMPORTANT TO READ! Giving x for derivative of sigmoid(x) is calculated before therefore no need to recalculate
    Args:
        z:  matrix that will calculate
        derive: check user wants to derive of sigmoid(x) function
    Returns:
        return output of sigmoid (x) function of its derivative
    """
    if( derive ):
        #derivative of sigmoid
        return z * ( 1  - z )
    #sigmoid function 
    return 1 / ( 1 + np.exp( -z ))

def softmax_activation(x,derive = False ):
    """
    This function finds softmax and its derivative 
    !IMPORTANT TO READ! Giving x for derivative of softmax(x) is calculated before therefore no need to recalculate
    Args:
        x: data
        derive: derivation of softmax given output of softmax
    Returns:
        return derivation of softmax or softmax itself 
    """      
    if derive:
        '''According to Jacobian matrix  row i and column j
        if i == j --> softmax * ( 1 - softmax )
        else --> -softmax * softmax
        '''
        dx = []
        length = x.shape[0]
        for i in range(length):

            col_soft = x[i,:].reshape(-1,1)
            result = np.diagflat(col_soft) - np.dot(col_soft, col_soft.T)
            dx.append(result)
        return np.array(dx)
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def dot_func(X, W, b ):
    """
    This function makes calculations this X.W + b
    Args:
        X: input matrix (m,n)
        W: weight matrix (n,p)
        b: bias matrix (1,p)
    Returns:
        return calculation results 
    """
    return np.dot(X,W) + b 

def mse(y, y_, derive = False ):
    """
    This function finds mean square error 
    Args:
        y: real label
        y_: prediction
        derive: check whether user wants to mse derivative
    Returns:
        return calculation results 
    """
    #length of labels
    length_of_data = y.shape[0]
    length_of_features = y.shape[1]
    if ( derive == True ):
        #derivation of mse
        return  ( 1.0 / (  length_of_data * length_of_features )  ) * np.sum(  np.subtract(y_, y), axis=0, keepdims= True  )
    #mse 
    return ( 1.0 / ( 2.0 * length_of_features * length_of_data )) * np.sum( np.square( np.subtract( y_, y)))

def next_batch(X, y,batch_size): 
    """
    create batches
    Args:
        X: data
        y: labels
        batch_size: batch size
    Returns:
        return batch 
    """      
    #make batch                                            
    for i in np.arange(0, X.shape[0], batch_size):
        #if smaller than X.shape[0]                                        
        if( i + batch_size <= X.shape[0] ):                            
            yield(X[i:i+batch_size], y[i:i+batch_size])
        #if bigger than X.shape[0]
        else:
            yield(X[i:X.shape[0]], y[i: X.shape[0]])







def backpropagation_q3_a(model, X, y, lr_rate ):
    """
    This function calculate backpropagation 4-a
    Args:
    
        model: has weights, biases and last prediction
        X: input
        y:labels
        lr_rate: learning rate
    Returns:
        return updated weights and biases
    """    
    #number of samples
    num_of_samples = 1/y.shape[0]
    dz2 = mse(y, model[2]['a'], derive = True ) * tanh_activation( model[2]['a'], derive = True )
    #gradient of weights
    dw2 = num_of_samples * np.dot(model[1]['a'].T, dz2)
    db2 = num_of_samples * np.sum(dz2, axis = 0, keepdims=True)
    dz1 = np.multiply(  np.dot(dz2, model[2]['w'].T), tanh_activation(model[1]['a'],derive=True))
    dw1 = num_of_samples * np.dot(X.T, dz1)
    db1 = num_of_samples * np.sum(dz1, axis = 0, keepdims= True)

    
   
    #update weights
    model[2]['w'] -= lr_rate * dw2
    model[2]['b'] -= lr_rate * db2
    model[1]['w'] -= lr_rate * dw1
    model[1]['b'] -= lr_rate * db1
                
    
    return 
    
def forward_propagation_q3(X, model, counter = 1 ):
    """
    Args:
        X : data
        model: the neural network model
        counter: counts number of layer
    Returns:
    return losses and accuracies
    """     
    #base statement
    if counter not in model:
        return 
    #forwar prop for one layer
    z = dot_func(X, model[counter]['w'], model[counter]['b'])  
    a = tanh_activation(z)
    model[counter]['a'] = a
    
   
    counter += 1 
    #recursive call
    return forward_propagation_q3(a, model, counter )

def q3_metrics(X, y, model):
    """
    Args:
       X: data
       y : labels
       model : neural network model
    Returns:
    return metrics
    """     
    #forward prop
    forward_propagation_q3(X, model )
    #check number of layer
    if 3 not in model:
        pred = model[2]['a'].copy()
    else :
        pred = model[3]['a'].copy()
    #accuracies    
    pred[ pred < 0 ] = -1
    pred[ pred > 0 ] = 1 
    result = y - pred 
    #loss
    mean_class_error = len(result[result == 0 ])/ len(result)
    mse_loss = mse(y, pred )
    
    return {'loss' : mse_loss, 'acc' : mean_class_error}
    
    
def q3_a(lr_rate=0.4,hidden_unit_size = 4, batch_size = 24, epoch =3000,title='Q3-A' ):
    """
    Args:
       
        lr_rate : learning rate
        hidden_unit_size1 : first hidden unit size
        hidden_unit_size2 : second hidden unit size 
        batch_size : mini batch size
        epoch : number of epoch
    Returns:
    return losses and accuracies
    """     
    
    #random shuffle train
    q3_a.X_train, q3_a.y_train = random_shuffle(q3_a.X_train, q3_a.y_train)
    #random shuffle test
    q3_a.X_test, q3_a.y_test = random_shuffle(q3_a.X_test, q3_a.y_test)
    #wieght and bias initializer
    
    firstlayer = xavier_weight_and_bias_initializer(w_size = (1024,hidden_unit_size), b_size = (1,hidden_unit_size))
    second_layer = xavier_weight_and_bias_initializer(w_size = (hidden_unit_size,1), b_size = (1,1))
    #make model 
    q3_a.model = { 1:firstlayer, 2: second_layer}
    #record loss
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for i in range(epoch):
        #record loss across one epochs.
        losses = []
        #random shuffle each epoch 
        q3_a.X_train, q3_a.y_train = random_shuffle(q3_a.X_train, q3_a.y_train)
        #create batch default batch_size = 100 
        for X_train_batch, y_train_batch in next_batch(q3_a.X_train, q3_a.y_train, batch_size ):
            #forward pass
            forward_propagation_q3(X_train_batch, q3_a.model)

            #record last loss
           
            #update weights
            backpropagation_q3_a(q3_a.model, X_train_batch, y_train_batch, lr_rate )
            #add loss lossess
            
        #record all losses and accuracies 
        train_metrics = q3_metrics(q3_a.X_train, q3_a.y_train, q3_a.model)
        val_metrics = q3_metrics(q3_a.X_test, q3_a.y_test, q3_a.model)
        train_loss.append(train_metrics['loss'])
        train_acc.append(train_metrics['acc'])
        val_loss.append(val_metrics['loss'])
        val_acc.append(val_metrics['acc'])
        
        #print    
        if( (i+1) % 100 == 0 ):
            print("Epoch ", i+1 , " ==> ", 'train_loss : {:.3f} || train_acc :{:.3f} || test_loss {:.3f} || test_acc : {:.3f}'\
                  .format(train_metrics['loss'],train_metrics['acc'],val_metrics['loss'], val_metrics['acc']))  
    plt.figure()
    plt.rcParams["figure.figsize"] = (16,12)
    plt.plot(range(epoch), train_loss, label = 'train_loss')#0.00001
    plt.plot(range(epoch), train_acc, label = 'train_acc')#0.00001
    plt.plot(range(epoch), val_loss, label = 'test_loss' )#0.00001
    plt.plot(range(epoch), val_acc, label = 'val_acc')#0.00001
    plt.legend()  
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.title(title)
    plt.show()
    return train_loss, train_acc, val_loss, val_acc

def backpropagation_q3_d(model, X, y, lr_rate, momentum_rate ):
    """
    This function calculate backpropagation 4-a
    Args:
    
        model: has weights, biases and last prediction
        X: input
        y:labels
        lr_rate: learning rate
        momentum_rate : momentum rate
    Returns:
        return updated weights and biases
    """    
    #number of samples
    num_of_samples = 1/y.shape[0]
    dz3 = mse(y, model[3]['a'], derive = True ) * tanh_activation( model[3]['a'], derive = True )
    #gradient of weights
    dw3 = num_of_samples * np.dot(model[2]['a'].T, dz3)
    db3 = num_of_samples * np.sum(dz3, axis = 0, keepdims=True)
    dz2 = np.multiply(  np.dot(dz3, model[3]['w'].T), tanh_activation(model[2]['a'],derive=True))
    dw2 = num_of_samples * np.dot(model[1]['a'].T, dz2)
    db2 = num_of_samples * np.sum(dz2, axis = 0, keepdims= True)
   
    dz1 = np.multiply(  np.dot(dz2, model[2]['w'].T), tanh_activation(model[1]['a'],derive=True))
    dw1 = num_of_samples * np.dot(X.T, dz1)
    db1 = num_of_samples * np.sum(dz1, axis = 0, keepdims= True)
    
    #add momentum variable
    model[3]['dw'] = momentum_rate * model[3]['dw'] - lr_rate * dw3
    model[3]['db'] = momentum_rate * model[3]['db'] - lr_rate * db3
    model[2]['dw'] = momentum_rate * model[2]['dw'] - lr_rate * dw2
    model[2]['db'] = momentum_rate * model[2]['db'] - lr_rate * db2
    model[1]['dw'] = momentum_rate * model[1]['dw'] - lr_rate * dw1
    model[1]['db'] = momentum_rate * model[1]['db'] - lr_rate * db1
    
    #update weights
    model[3]['w'] += model[3]['dw']
    model[3]['b'] += model[3]['db']
    model[2]['w'] += model[2]['dw']
    model[2]['b'] += model[2]['db']
    model[1]['w'] += model[1]['dw']
    model[1]['b'] += model[1]['db']
   
    
    
    return 
    
    
def q3_d(momentum_rate = 0.0, lr_rate=0.25,hidden_unit_size1 = 32 ,hidden_unit_size2 =8, batch_size = 32, epoch =1200, title="Q3-D" ):
    """
        Args:
            momentum_rate : momentum rate
            lr_rate : learning rate
            hidden_unit_size1 : first hidden unit size
            hidden_unit_size2 : second hidden unit size 
            batch_size : mini batch size
            epoch : number of epoch
        Returns:
        return losses and accuracies
    """     
    #random shuffle train
    q3_a.X_train, q3_a.y_train = random_shuffle(q3_a.X_train, q3_a.y_train)
    #random shuffle test
    q3_a.X_test, q3_a.y_test = random_shuffle(q3_a.X_test, q3_a.y_test)
    #wieght and bias initializer
    
    firstlayer = xavier_weight_and_bias_initializer(w_size = (1024,hidden_unit_size1), b_size = (1,hidden_unit_size1))
    second_layer = xavier_weight_and_bias_initializer(w_size = (hidden_unit_size1, hidden_unit_size2), b_size = (1,hidden_unit_size2 ))
    third_layer = xavier_weight_and_bias_initializer(w_size = (hidden_unit_size2,1), b_size = (1,1))
    
    #make model 
    q3_d.model = { 1:firstlayer, 2: second_layer, 3 : third_layer}
    #initial gradients
    for i in range(1, 4):
        q3_d.model[i]['db'] = q3_d.model[i]['b'] * 0 
        q3_d.model[i]['dw'] = q3_d.model[i]['w'] * 0 
    #record loss
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for i in range(epoch):
        
        #random shuffle each epoch 
        q3_a.X_train, q3_a.y_train = random_shuffle(q3_a.X_train, q3_a.y_train)
        #create batch default batch_size = 100 
        for X_train_batch, y_train_batch in next_batch(q3_a.X_train, q3_a.y_train, batch_size ):
            #forward pass
            forward_propagation_q3(X_train_batch, q3_d.model)

            #record last loss
           
            #update weights
            backpropagation_q3_d(q3_d.model, X_train_batch, y_train_batch, lr_rate, momentum_rate  )
            #add loss lossess
            
        #record all losses and accuracies 
        train_metrics = q3_metrics(q3_a.X_train, q3_a.y_train, q3_d.model)
        val_metrics = q3_metrics(q3_a.X_test, q3_a.y_test, q3_d.model)
        train_loss.append(train_metrics['loss'])
        train_acc.append(train_metrics['acc'])
        val_loss.append(val_metrics['loss'])
        val_acc.append(val_metrics['acc'])
        
        #print    
        if( (i+1) % 100 == 0 ):
            print("Epoch ", i+1 , " ==> ", 'train_loss : {:.3f} || train_acc :{:.3f} || test_loss {:.3f} || test_acc : {:.3f}'\
                  .format(train_metrics['loss'],train_metrics['acc'],val_metrics['loss'], val_metrics['acc']))  
    plt.figure()
    plt.rcParams["figure.figsize"] = (16,12)
    plt.plot(range(epoch), train_loss, label = 'train_loss')
    plt.plot(range(epoch), train_acc, label = 'train_acc')
    plt.plot(range(epoch), val_loss, label = 'test_loss' )
    plt.plot(range(epoch), val_acc, label = 'val_acc')
    plt.legend()  
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.title(title)
    plt.show()
    return train_loss, train_acc, val_loss, val_acc
def plot_conf_matrix(cm, classes, title='Confusion matrix',cmap=plt.cm.Blues):
    '''
    Plots confusion matrix, heavily inspired from scikit.learn website
    '''
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j],  'd'),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
class Neural(object):
    'Neural Network for 14'
    def __init__(self, lambdax = 0 ):
        '''
        construct
        '''
        first =  glorot_weight_and_bias_initializer(w_size=(784,40), b_size=(1,40))
        second =  glorot_weight_and_bias_initializer(w_size=(40,26), b_size=(1,26))
        self.lambdax = lambdax
        self.w1 = first['w']
        self.b1 = first['b']
        self.w2 = second['w']
        self.b2 = second['b']
    def forward_prop(self,X):
        '''
        return predictions
        '''
        self.z1 = np.dot(X, self.w1.reshape(784,40)) + self.b1.reshape(1,40)
        self.a1 = sigmoid_activation(self.z1)
        self.z2 = np.dot(self.a1, self.w2.reshape(40,26)) + self.b2.reshape(1,26)
        self.a2 = softmax_activation(self.z2)
      
        return self.a2
        
        
    def costFunction(self,X,y):
        '''
        return cost function of q4
        '''
        m = y.shape[0]
        n = y.shape[1]
   
        self.yHat = self.forward_prop(X)
        log_likelihood = -np.log(self.yHat[range(m),y])
        J = (np.sum(log_likelihood) + (self.lambdax/2)*(np.sum(np.square(self.w1)) + np.sum(np.square(self.w2)))) /( m * n ) 
        #print("Loss : ", J)
        return J
    
    def costFunctionPrime(self, X, y ): 
        '''
        returns derivative of q4
        '''
   
        m = y.shape[0]
        n = y.shape[1]
        self.dz2 = self.forward_prop(X)
     
        self.dz2[range(m),y] -= 1
        
        
        self.dw2 =   (np.dot(self.a1.T, self.dz2) + self.lambdax * self.w2.reshape(40,26)) /m
        self.db2 =  np.sum(self.dz2, axis = 0, keepdims=True)  /m
        self.dz1 = np.multiply(  np.dot(self.dz2, self.w2.reshape(40,26).T), sigmoid_activation(self.a1,derive=True))
        self.dw1 =  (np.dot(X.T, self.dz1) + self.lambdax * self.w1.reshape(784,40))/m 
        self.db1 = np.sum(self.dz1, axis = 0, keepdims= True)  /m
        
        return self.dw1, self.db1, self.dw2, self.db2
    def setParams(self,params):
        '''
        set weights
        '''
        self.w1 = params[:784*40].reshape(784,40)
        self.b1 = params[784*40:784*40+40]
        self.w2 = params[784*40+40:(784*40)+40+(40*26)]
        self.b2 = params[(784*40)+40+(40*26):]
        
    def getParams(self):
        '''
        get weights
        '''
        params = np.concatenate((self.w1.ravel(), self.b1.ravel(), self.w2.ravel(), self.b2.ravel()))
      
        return params
    
    def giveGradients(self,X,y):
        '''
        returns gradients
        '''
        self.dw1, self.db1, self.dw2, self.b2 = self.costFunctionPrime(X,y)
        return np.concatenate((self.dw1.ravel(), self.db1.ravel(), self.dw2.ravel(), self.db2.ravel()))

    
        
class Train(object):
    '''
    conjugate gradient training class
    '''
    def __init__(self,N):
        'construct'
        self.N = N
        
    def callback(self,params):
        '''
        set weights
        '''
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))
    
    def costFunctionWrap(self, params, *args):
        '''
        send cost and grad 
        '''
        X,y = args
        self.N.setParams(params)
        loss = self.N.costFunction(X,y )
        gradx = self.N.giveGradients(X,y)
        return loss, gradx
    
    def train(self, X, y):
        
        self.X = X 
        self.y = y 
        
        self.J = []
        
        params0 = self.N.getParams()
        
        options = {'maxiter' : 50, 'disp' : True }
        
        _res = optimize.minimize(self.costFunctionWrap ,params0, jac=True,method='CG',\
                                args=(X,y), options=options, callback =self.callback)
        self.N.setParams(_res.x)
        self.optimizationResults = _res

def q1():
    """
    q1 implementation
   
    """     
    def q1_plot(x_train,y_train, x_test, w, b, title  ):
        """
        plot decision boundary
        Args:
            x_train : train data
            y_train: train label
            x_test: test data
            w: weight
            b:bias
            title:title of plot
     
        """     
        plt.figure()
        
        inputs = x_train
        inputs_x = x_test
        targets = y_train
        
        for i in range(2):
            plt.plot(inputs_x[i,0],inputs_x[i,1], 'bd')
            
        plt.plot(inputs_x[2,0], inputs_x[2,1], 'rd', label='test data')
        
        for input,target in zip(inputs,targets):
            plt.plot(input[0],input[1],'ro' if (target == 1.0) else 'bo')
            
        plt.plot(inputs[-1,0], inputs[-1,1],'ro' if (target == 1.0) else 'bo', label = 'train data' )
        #decision boundary
        x = np.linspace(np.amin(inputs[:,:1]),np.amax(inputs[:,:1])).reshape(1,50)
        y = (-w[0]/w[1])* x + b
        
        #draw
        plt.grid(True)
        plt.legend()
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(title)
        plt.plot(x, y , 'k.')  
        plt.show()
    
    def q1_perceptron(x_train,y_train, w_in,b_in):
        """
        Args:
            x_train : train data
            y_train: train label
            w_in : initial weights
            b_in : initial bias
        Returns:
        return updated weight and bias
        """     
        w = w_in.copy()
        b = b_in.copy()
        #hold losses
        losses = []
        count = 0 
        for i in range(10):
            for X,y in next_batch(x_train, y_train, 1):

                count += 1
                #forward prop
                z1 = dot_func(X, w, b )
                a1 = tanh_activation(z1)
                probs = a1.copy()
                loss = (mse(y, probs))
                losses.append(loss)
                #back prop
                dz = mse(y, a1, derive = True ) * tanh_activation(a1, derive = True)
                dw =  ( np.dot(X.T, dz) ) 
                db =  np.sum(dz, axis = 0, keepdims=True)
                print('count :', count ,'\ndw : ', dw, '\ndb : ', db)
                #update
                w  -= dw
                b  -= db
        return w, b
    
    def q1_hessian(x_train, y_train, w_in,b_in ):
        """
        Args:
            x_train : train data
            y_train: train label
            w_in : initial weights
            b_in : initial bias
        Returns:
        return updated weight and bias
        """     
        w = w_in.copy()
        b = b_in.copy()
        for i in range(4):
            for j in range(2):
                w[j] = w[j]+ y_train[i]*x_train[i,j]
            b += y_train[i]
        return w, b 
    #datas
    x_train = np.array([[2,-1],[1,4],[0,-2], [ -2,3 ]]).astype(np.float64)
    y_train = np.array([-1,-1,1,1]).astype(np.float64).reshape(4,1)
    x_test = np.array([[1,0], [0,0], [-1.5,3]])
    #initial weights
    w = np.array([ 0 , 0 ]).astype(np.float64).reshape(2,1)
    b = np.array([ 0 ]).astype(np.float64).reshape(1,1)
    #updated weights
    w_p, b_p = q1_perceptron(x_train, y_train, w,b )
    w_h, b_h = q1_hessian(x_train, y_train,w,b)
    #plots
    q1_plot(x_train, y_train, x_test, w_p, b_p, 'PERCEPTRON')
    q1_plot(x_train, y_train, x_test, w_h, b_h, 'HEBBIAN')
    
def q3():
    #a
    print('\n3-a loading...\n')
    q3_a()
    #c 
    #low
    print('\n3-c loading...\n')
    q3_a(hidden_unit_size=1, title='Q3-C-low')
    q3_a(hidden_unit_size=4, title='Q3-C-optimal')
    q3_a(hidden_unit_size=64, title='Q3-C-high')
    #d 
    print('\n3-d loading...\n')
    q3_d()
    print('\n3-e loading...\n')
    q3_d(momentum_rate=0.5, title= 'Q3-E-Momentum 0.5')
    
def q4(lambdax=0):
    X_train = q4.X_train.copy()
    #make preprocess train labels 
    y_train = q4.y_train.copy()

    #subtract one to find true classes
   
    NN = Neural(lambdax=lambdax)
    a = NN.getParams()
    X_train, y_train = random_shuffle(X_train,y_train)
    T = Train(NN)
    for X, y in next_batch(X_train, y_train,1):
        T.train(X,y)
    
    
    a = NN.forward_prop( q4.X_train)

    pred = np.argmax(a, axis=1).reshape(-1,1)
    
    cnf_matrix = confusion_matrix(q4.y_train, pred)
    class_names = list(string.ascii_lowercase)
    plot_conf_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix Train')
    
    a = NN.forward_prop( q4.X_test)

    pred = np.argmax(a, axis=1).reshape(-1,1)
    
    cnf_matrix = confusion_matrix(q4.y_test, pred)
    class_names = list(string.ascii_lowercase)
    plot_conf_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix Test')
    b = NN.getParams().copy()
    b = b[:784*40].reshape(784,40)
    plt.figure()
    for i in range(0,40):
        plt.subplot(4,10,i+1)
        plt.rcParams["figure.figsize"] = (10,8)
        weight_image = b[:,i].copy()

        weight_image = (weight_image *255).astype(np.int) 
        weight_image = weight_image.reshape(28,28)
        plt.imshow(weight_image, cmap = plt.get_cmap('gray'))
        #plt.imsave('image.jpg', weight_image, cmap='gray')
        #plt.figure(figsize=(3,3))
        plt.imsave('image2-{}.png'.format(i), weight_image, cmap = plt.get_cmap('gray'))
    plt.show()
    
#make preprocess train images 
q3_a.X_train = img_data_generator(data['trainims'])
#make preprocess train labels 
q3_a.y_train = labels_generator(data['trainlbls'],polarity = 'bipolar')
q3_a.X_test = img_data_generator(data['testims'])
q3_a.y_test = labels_generator(data['testlbls'], polarity='bipolar')

#q4 model has weight q4.model['w1'] and biases q4.model['b1']
q3_a.model = {}
q3_d.model = {}
#make preprocess train images 
q4.X_train = img_data_generator(data2['trainims'])

y_temp = labels_generator(data2['trainlbls'], polarity='unipolar')
y_copy = y_temp.copy()
q4.y_train = y_copy - np.ones((5200,1))
q4.y_train = q4.y_train.astype(np.int)

q4.X_test = img_data_generator(data2['testims'])
y_temp = labels_generator(data2['testlbls'], polarity='unipolar')
y_copy = y_temp.copy()
q4.y_test = y_copy - np.ones((1300,1))
q4.y_test = q4.y_test.astype(np.int)


question = sys.argv[1]

def sait_akturk_21501734_hw2(question):
    if question == '1' :
        print (question)
        q1()
    elif question == '3' :
        print (question)
        q3()
    elif question == '4' :
        print (question)
        q4()
sait_akturk_21501734_hw2(question)