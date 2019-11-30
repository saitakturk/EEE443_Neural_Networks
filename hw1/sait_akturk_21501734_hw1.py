import numpy as np
import scipy.io
from sklearn.metrics import classification_report
import matplotlib.animation as animation
import matplotlib.pyplot as plt 
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
    #initialize weights
    W = np.random.normal(size = w_size, loc = loc, scale = scale ).astype(np.float64)
    #initialize biases
    b = np.random.normal(size = b_size, loc = loc , scale = scale ).astype(np.float64)
    return W, b  

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
    e_x = np.exp(x - np.max(x))
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
    return ( 1.0 / ( 2.0 *  length_of_data  *  length_of_features )) * np.sum( np.square( np.subtract( y_, y)))

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



def one_hot_encoder( y, class_size ):
    """
    This function makes one hot encode
    Args:
    
        y: labels
        class_size: number of classes 
    Returns:
        return one hot vectors of labels
    """      
    y_temp = []
    for y_sample in y:
        #make zero vector size of class size 
        one_hot_sample = np.zeros([class_size])
        #make 1 specificed labels
        one_hot_sample[int(y_sample-1)] = 1
        #add the bag
        y_temp.append(one_hot_sample)
    #make numpy array 
    y_temp = np.array(y_temp)
    return y_temp

def backpropagation_q4(model, X, y, lr_rate ):
    """
    This function makes one hot encode
    Args:
    
        model: has weights, biases and last prediction
        X: input
        y:labels
        lr_rate: learning rate
    Returns:
        return updated weights and biases
    """    
    #weights
    w1 = model['w1']
    #biases
    b1 = model['b1']
    #prediction
    pred = model['pred']
    #delta
    dz = mse(y, pred, derive = True ) * tanh_activation( pred, derive = True )
    #gradient of weights
    dw1 = np.dot(X.T, dz)
    #gradient of biases
    db1 = np.sum(dz, axis=0, keepdims = True)
    #update weights and biases
    w1 -= lr_rate * dw1
    b1 -= lr_rate * db1
    
    return w1, b1 

def backpropagation_q5(model, X, y, lr_rate):
    """
    backprop  for q5
    Args:
    
        model: has weights, biases and last prediction
        X: input
        y:labels
        lr_rate: learning rate
    Returns:
        return updated weights and biases
    """    
    #mse derivative [1,n]
    d_mse = mse(y, model['pred'],  derive=True)
    #softmax derivative [n,n]
    d_softmax = softmax_activation(model['pred'],derive= True )
    #delta [1,n]
    dz = np.matmul(d_mse, d_softmax).squeeze(axis=1)
    #gradient of weights and biases 
    dw = np.dot(X.T, dz )
    db = np.sum(dz, axis=0, keepdims = True)
    #update weights and biases 
    model['w1'] -= lr_rate * dw
    model['b1'] -= lr_rate * db
    
    return model['w1'], model['b1']


def q3_forward(X,W,w_out,y):
    """
    q3 forward
    Args:
    
        model: has weights, biases and last prediction
        X: input
        W:labels
        w_out: learning rate
        y:labels
    Returns:
        return accuracy
    """  
    z = np.dot(X, W )
    #step activation function [m,n]
    z[ z < 0 ] = -100
    z[ z >= 0 ] = 1 
    z[ z == -100 ] = 0
    #add biases multiplier 1 [m,n+1]
    c = np.ones( (z.shape[0], z.shape[1]+1))
    #put data 
    c[:,:-1] = z
    
    z2 = np.dot(c, w_out)
    #step activation function
    z2[ z2 < 0 ] = -100
    z2[ z2 >= 0 ] = 1  
    z2[ z2 == -100 ] = 0 
    z2 = z2.astype(np.int)
    result = y == z2
    return len(result[result == 1 ]) / (y.shape[0] * y.shape[1])

def q3():
    #X input as column represents x1 x2 x3 x4 1, row represents number of training samples
    
    X = [[0,0,0,0,1],
         [0,0,0,1,1],
         [0,0,1,0,1],
         [0,0,1,1,1],
         [0,1,0,0,1],
         [0,1,0,1,1],
         [0,1,1,0,1],
         [0,1,1,1,1],
         [1,0,0,0,1],
         [1,0,0,1,1],
         [1,0,1,0,1],
         [1,0,1,1,1],
         [1,1,0,0,1],
         [1,1,0,1,1],
         [1,1,1,0,1],
         [1,1,1,1,1]]
    #output 
    y = [[0],
         [0],
         [0],
         [1],
         [1],
         [1],
         [1],
         [0],
         [0],
         [0],
         [0],
         [1],
         [0],
         [0],
         [0],
         [1]]

    X = np.array(X)
    y = np.array(y)
    #weight vector of hidden units as column h1 h2 h3 h4, rows are w1 w2 w3 w4 w0
    W = [[0, -1, 1, -1],
         [-1, 1, 0, 1],
         [1, -1, 1, 0],
         [1, 0, 1, -1],
         [-1.5, -0.5, -2.5, -0.5]]
    W = np.array(W)
    #output neuron weigths as rows w1,w2,w3,w4,w0
    w_out = [[1],
             [1],
             [1], 
             [1],
             [-0.5]]
    w_out= np.array(w_out)

    #Part B 
    print("\n***Q3-Part{}***\n".format("B"))
    print("Accuracy : ", q3_forward(X,W,w_out,y, ))
    #Part C
    print("\n***Q3-Part{}***\n".format("C"))
    small_noise = np.random.normal(size= (16,4), loc=0, scale= 0.1).astype(np.float64)
    X_small_noise = X.copy().astype(np.float64)
    X_small_noise[:,:-1] += small_noise
    print("Accuracy : ", q3_forward(X_small_noise ,W,w_out,y, ))
    #Part D 
    print("\n***Q3-Part{}***\n".format("D"))
    w_out_new = np.array([[ 0.86936658],
                          [ 0.86936658],
                          [ 0.86936658],
                          [ 0.86936658],
                          [-0.71394658]])

    W_new = np.array([[-0.0157257 , -1.01352933,  0.9810651 , -1.01377487],
                      [-1.01299569,  0.9832329 , -0.0169859 ,  0.98346286],
                      [ 0.98026133, -1.01528407,  0.98095781, -0.02154296],
                      [ 0.98344734, -0.01811416,  0.98318884, -1.01240249],
                      [-1.53866706, -0.53777454, -2.53547764, -0.53902452]])
    X_hundred = [] 
    y_hundred = []
    random_hundred = np.floor(np.random.uniform(0,16,100)).astype(np.int) 
    for i in random_hundred:
        X_hundred.append(X[i,:])
        y_hundred.append(y[i])
    X_hundred = np.array(X_hundred)
    y_hundred = np.array(y_hundred)
    noise = np.random.normal(size =(100,4), loc = 0, scale= 0.2)
    X_hundred_noise = X_hundred.copy().astype(np.float64)
    X_hundred_noise[:,:-1] += noise 
    print("Old weights accuracy on d part : ", q3_forward(X_hundred_noise, W, w_out, y_hundred))
    print("New weights accuracy on d part : ",q3_forward(X_hundred_noise, W_new, w_out_new, y_hundred))
    print("New weights accuracy on c part : ", q3_forward(X_small_noise, W_new, w_out_new, y))
    print("New weights accuracy on b part : ", q3_forward(X, W_new, w_out_new, y))
    return
def q4_test():
    """
    This function test q4, you should process q4(), otherwise program will crush
    Args:
    Returns:
    """  
    z = dot_func(q4.X_test, q4.model['w1'], q4.model['b1'])
    a = tanh_activation(z)
    a[ a < 0 ] = -1 
    a[ a>= 0 ] = 1
    print(classification_report(y_pred = a, y_true = q4.y_test, target_names={'cat', 'car'} ))
    return 

def q4(lr_rate=1e-3,batch_size = 100, epoch =10000):
    """
    This is q4 solution
    Args:
    
        lr_rate: learning rate default 1e-3
        batch_size: batch size default 100
        epoch: number of epochs default 10000
    
    Returns:
        return recorded losses
    """  
    print("Q4 is loading...")
    #random shuffle train
    q4.X_train, q4.y_train = random_shuffle(q4.X_train, q4.y_train)
    #random shuffle test
    q4.X_test, q4.y_test = random_shuffle(q4.X_test, q4.y_test)
    #wieght and bias initializer
    w1, b1 = normal_weight_and_bias_initializer(w_size = (1024,1), b_size = (1,1),loc = 0.0, scale = 0.01)
    #make model 
    q4.model = {'w1': w1, 'b1': b1}
    #record loss
    train_loss = []
    for i in range(epoch):
        #record loss across one epochs.
        losses = []
        #random shuffle each epoch 
        q4.X_train, q4.y_train = random_shuffle(q4.X_train, q4.y_train)
        #create batch default batch_size = 100 
        for X_train_batch, y_train_batch in next_batch(q4.X_train, q4.y_train, batch_size ):
            #forward pass
            z = dot_func(X_train_batch,q4.model['w1'], q4.model['b1'])
            pred = tanh_activation(z)
            #record predictions or output 
            q4.model['pred'] = pred 
            #MSE Loss
            loss = mse(y_train_batch, pred )
            #record last loss
            q4.model['loss'] = loss
            #update weights
            q4.model['w1'], q4.model['b1'] = backpropagation_q4(q4.model, X_train_batch, y_train_batch, lr_rate )
            #add loss lossess
            losses.append(loss) 
        #print    
        if( (i+1) % 1000 == 0 ):
            print("Epoch ", i+1 , " : ", np.mean(losses), 'lr rate : ', lr_rate)  
        #record loss all over epochs
        train_loss.append(np.mean(losses))
        q4.ax1.clear()
        q4.ax1.plot(range(i+1), train_loss)
    return np.array(train_loss)



def q5_test():
    """
    This function test q4, you should process q4(), otherwise program will crush
    Args:
    Returns:
    """  
        
    z = dot_func(q5.X_test, q5.model['w1'], q5.model['b1'])
    a = softmax_activation(z)
    a = np.argmax(a, axis= 1)
    #labels
    labels = list(string.ascii_lowercase)
    c = np.argmax(q5.y_test_one_hot, axis = 1 )
    report = classification_report(y_pred=a , y_true=c, target_names=labels)
    print(report )
    return 
	
	
	
def q5(lr_rate=0.5, batch_size = 100, epoch=10000):
    """
    This is q5 solution
    Args:
    
        lr_rate: learning rate default 0.5
        batch_size: batch size default 100
        epoch: number of epochs default 10000
    
    Returns:
        return recorded losses
    """  
    print("Q5 is loading...")
    #random shuffle train
    q5.X_train5, q5.y_train5 = random_shuffle(q5.X_train, q5.y_train)
    #make one hot vector labels 
    q5.y_train_one_hot = one_hot_encoder(q5.y_train,26)

    #random shuffle test
    q5.X_test, q5.y_test = random_shuffle(q5.X_test, q5.y_test)
    #make one hot vector labels 
    q5.y_test_one_hot = one_hot_encoder(q5.y_test,26)
    
    #initialize weights and biases
    w1, b1 = normal_weight_and_bias_initializer(w_size = (784,26), b_size = (1,26), loc = 0.0, scale = 0.01)
    #create model
    q5.model = {'w1': w1, 'b1': b1}
    #record losses
    train_loss = []


    for i in range(epoch):
        losses = []
        #random shuffle
        q5.X_train, q5.y_train = random_shuffle(q5.X_train, q5.y_train)
        #make one hot vector 
        q5.y_train_one_hot = one_hot_encoder(q5.y_train,26)
        #one epoch
        for X_train_batch, y_train_batch in next_batch(q5.X_train, q5.y_train_one_hot,  batch_size ):
            #forward pass
            z = dot_func(X_train_batch,q5.model['w1'], q5.model['b1'])
            pred = softmax_activation(z)
            #prediction record
            q5.model['pred'] = pred 
            #loss
            loss = mse(y_train_batch, pred )
            #record loss
            q5.model['loss'] = loss
            #update model
            q5.model['w1'], q5.model['b1'] = backpropagation_q5(q5.model, X_train_batch, y_train_batch, lr_rate )
            #record loss
            losses.append(loss) 
        #print results
        if( i == 0  or (i+1) % 100 == 0 ):

            print("Epoch ", i+1 , " : ", np.mean(losses), 'lr rate : ', lr_rate)
        #record epoch lossess
        train_loss.append(np.mean(losses))
    q5_test()
    return np.array(train_loss)




#make preprocess train images 
q4.X_train = img_data_generator(data['trainims'])
#make preprocess train labels 
q4.y_train = labels_generator(data['trainlbls'],polarity = 'bipolar')

#make preprocess test images
q4.X_test = img_data_generator(data['testims'])
#make preprocess tes labels
q4.y_test = labels_generator(data['testlbls'], polarity = 'bipolar')
#q4 model has weight q4.model['w1'] and biases q4.model['b1']
q4.model = {}



#make preprocess train images 
q5.X_train = img_data_generator(data2['trainims'])
#make preprocess train labels 
q5.y_train = labels_generator(data2['trainlbls'],polarity = 'unipolar')
#make one hot vector labels 
q5.y_train_one_hot = one_hot_encoder(q5.y_train,26)
#make preprocess test images
q5.X_test = img_data_generator(data2['testims'])
#make preprocess test labels 
q5.y_test = labels_generator(data2['testlbls'], polarity = 'unipolar')
#make one hot vector labels for test 
q5.y_test_one_hot = one_hot_encoder(q5.y_test,26)
#q5 model has weight q5.model['w1'] and biases q5.model['b1']
q5.model = {}



question = sys.argv[1]

def sait_akturk_21501734_hw1(question):
    if question == '3' :
        q3()
    elif question == '4' :
        print (question)
        ani = animation.FuncAnimation(fig, q4, interval=1000)
        plt.show()
    elif question == '5' :
        print (question)
        q5()
sait_akturk_21501734_hw1(question)