'''
%THIS IS THE SOLUTION FOR QUESTION 2.YOU SHOULD COPY THIS CODE AND PASTE PROPER .M FILE AND RUN IN MATLAB( COPY ONLY CODES IN MULTI LINE COMMENT)
%===============================Q2 MATLAB======================================================================
function sait_akturk_21501734_hw3(question)
clc
close all

switch question
    case '2'
	disp('2');
        %% question 2 code goes here
	q2();
           
end

end




function q2()
%Main function to run q2 for assignment 3 

%load image patches
[d, invX, x] = load_data();

%normalized image patches
normalized_data2 = normalize_patches(d);
%show all patches and normalized patches
show_images(d, normalized_data2);


%q2-2
%optimal hyper parameters
P.Lin = 256; %input layer dimension 
P.Lhid = 64; %hidden layer dimension
options.MaxIter = 200;
P.lambda = 5 * 1e-4;
P.rho= 0.010;
P.beta = 2.5; 



%weight and bias initialization
W_input = weight_and_bias_init(P.Lin, P.Lhid); 

%cost and gradients
J  = @(W_input) aeCost(W_input, normalized_data2, P);

%get updated weights
[Weights, fx, i]= fmincgrad(J, W_input,options);

%reshape w1 to ideal 
W1 = reshape(Weights(1 : P.Lhid*P.Lin), P.Lhid, P.Lin);

%show weights
show_weights(W1');


end 

function [ normalized ] = normalize_patches(d)
%This function normalize patches according to assignment 
%Params:
%   d : all image patches
%Return:
%   normalized : normalized image patches

%rgb to gray patches
post_d = gray_data(d);

%flatten
flat_post_d = flatten(post_d);

%remove the mean of all patches from image patches
normalized_d = find_mean_insensity(flat_post_d);



%divide std
stdx = 3 * std(normalized_d(:));
mean = max(min(normalized_d, stdx), -stdx) / stdx;

%make range [0.1, 0.9]
normalized = (mean + 1) * 0.4 + 0.1;

end 
function [weights ] = weight_and_bias_init(in, hid )
%This function initialize weights and biases
%Params:
%   in : input size 
%   hid : hidden size 
%Return:
%   weights : weights and biases 1D

%initialize weights 
w0  = sqrt(6) / sqrt(in+hid+1);   
w1 = rand(hid, in) * 2 * w0 - w0;
w2 = rand(in, hid) * 2 * w0 - w0;
%initialize biases
b1 =  rand(hid, 1) * 2 * w0 - w0;
b2 =  rand(in, 1) * 2 * w0 - w0;

weights = [ w1(:); w2(:); b1(:); b2(:)];

end 

function  show_images(d, normalized_data2)
%This function show patches and normalized patches
%Params:
%   d : all image patches
%   normalize_data2 : normalized image patches
%Return:
%   

%reshape data to plot
reshaped_data2 = reshape(normalized_data2, [ 16 , 16 ,10240 ] );

%empty matrix to show image patches 
rgb_images = zeros( 10 * 16, 20 * 16 ,3 );
%empty matrix to show normalized image patches 
normalized_images = zeros( 10 * 16, 20 * 16 ) ; 

%random image patches 
for i = 1:10
    for j = 1:20  
       rgb_images(((i-1) * 16 )+1 :i * 16, ((j-1) * 16 )+1 :j * 16, :) = d(:,:,:,((i-1)*20 + j)+200 );
       normalized_images(((i-1) * 16 )+1 :i * 16, ((j-1) * 16 )+1 :j * 16) = reshaped_data2(:,:,((i-1)*20 + j)+200);
    end
end

%plot random image patches
figure;
subplot(2,1,1)
imshow(rgb_images);
title('rgb patches');
subplot(2,1,2)
imshow(normalized_images);
title('normalized patches');



end

function mean_insensities = find_mean_insensity(flat_post_d)
%This function remove mean from image patches
%Params:
%   flat_post_d : all flatten gray image patches
%Return:
%   mean_insensities : removed mean image patches

%remove image mean
mean_insensities = bsxfun(@minus, flat_post_d, mean(flat_post_d));

end
function flat_post_d = flatten(post_d)
%This function flatten data format
%Params:
%   post_d  : image patches
%Return:
%   flat_post_d : flatten image patches

%size of array
flat_post_d = reshape(post_d,[256, 10240] );

end

function post_d = gray_data(d)
%This function conver rgb to gray
%Params:
%   d : all image patches
%Return:
%   post_d : all gray image patches
post_d =  0.2126 * d(:,:,1,:) + 0.7152 * d(:,:,2,:) + 0.0722 * d(:,:,2,:);


end 

function [d, invXForm, xForm ] = load_data()
%read the data to struct
data2 = load('assign3_data2.mat');
%convert data types 
data2_data = struct2cell( data2(1) );

%size 16x16x3x10240
d = cell2mat( data2_data(1) );
invXForm = cell2mat(data2_data(2));
xForm = cell2mat( data2_data(3));
end

function [h, array] = show_weights(weights)





%  rescaling for constrast adjusting
weights = weights - mean(weights(:));

% find rows and columns
[L number_of_filter] = size(weights);

%weight height default 16 
weight_height = sqrt(L);

%divisor of number of filters 
div = divisors(number_of_filter);

%find size of divisor number
[temp, no_of_div] = size(div);

%middle divisor to shape overall image
image_height = div(round(no_of_div/2));

%image width to show weight images
image_width = number_of_filter / image_height;

%empty aray to show all weigths
patches = zeros(image_height*(weight_height), image_height*(weight_height));



%count weights number
count = 1;
%set weights
for j = 1 : image_height
    for i = 1 : image_width
        if count > number_of_filter
            continue; 
        end
        %normalize weights to adjust constrast
        patches((i-1)*(weight_height)+(1:weight_height), (j-1)*(weight_height)+(1:weight_height)) = reshape(weights(:, count), weight_height,weight_height) / max(abs(weights(:, count)));
      
        count = count + 1;
    end
end
%plotweights
figure;
w = imagesc(patches, [-1 1]);
colormap(gray);
axis image off

drawnow;

end



'''

#*************************************************************Q1 PYTHON CODE BELOW*********************************************************************************************



import numpy as np

import h5py
from sklearn.manifold import TSNE 
#from adjustText import adjust_text
import matplotlib.pyplot as plt 
import os 
import sys
def read_data(batch_size):
    """
    This function read data for question 1 
    
    Args:
        batch_size : batch size for mini batch learning
    Returns:
        all data
    """

    data1 = {}
    #read all data
    with h5py.File('assign3_data1.mat', 'r') as file:

        testd = file['testd'][:,:].T
        testx = file['testx'][:,:].T
        traind = file['traind'][:,:].T
        trainx = file['trainx'][:].T
        vald = file['vald'][:].T
        valx = file['valx'][:].T
        words = [] 
        for i in range(250):
            L = file[file['words'][i].item()][:].reshape(-1)
            string = ''.join(map(chr,L))
            words.append(string)

        data1['words'] = words
        data1['valx'] = valx
        data1['vald'] = vald
        data1['trainx'] = trainx
        data1['traind'] = traind
        data1['testx'] = testx
        data1['testd'] = testd
        
        #number of batches 
        num_batches = np.int(data1['trainx'].shape[1] / batch_size )
        
        #subtract one for array index
        X_train = data1['trainx'][:,:num_batches * batch_size].reshape(3, batch_size, num_batches ) -1 
        y_train = data1['traind'][:,:num_batches * batch_size].reshape(1, batch_size, num_batches ) - 1
        X_val = data1['valx'] -1 
        y_val = data1['vald'].reshape(1,-1) -1 
        X_test = data1['testx'] - 1
        y_test = data1['testd'].reshape(1,-1) -1 
        
        words = np.array(data1['words']).reshape(1,-1)



    return X_train, y_train, X_val, y_val, X_test, y_test, words

def normal_weight_and_bias_initializer( w_size , b_size  , loc = 0.0, scale = 0.01):
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
    return W,  b  

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
    batch_idx = np.random.permutation(y.shape[1])
    X,y  = X[:,batch_idx, :], y[:, batch_idx, :] 
    sec_idx = np.random.permutation(y.shape[2])
    return  X[:,:, sec_idx], y[:, :, sec_idx] 

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
def cross_entropy(y, y_):
    """
    This function calculate cross entropy loss 
    
    Args:
        y : target
        y_ : prediction

    Returns:
        cross entropy loss
    """
   
    return -np.sum(np.sum(y* np.log(y_+np.exp(-32))))/y_.shape[1]


def backpropagation_q1(X, y, momentum_rate, lr_rate, model):
    """
    This function do backpropagation for q1
    
    Args:
        X:  data
        y: labels
        momentum_rate : momentum rate default 0.85
        lr_rate : learning rate default 0.15
        model : model that holds all data about network

    Returns:

    """
    #find batch size to find mean 
    batch_size = model['pred'].shape[1]
    
    #number of hidden unit 1
    num_hidden_unit1 = model['embed'].shape[1]
    #number of hidden unit in second hidden layer
    num_hidden_unit2 = model['w1'].shape[1]
    
    #error delta
    dz2 = model['pred'] - y
    
    #gradient of second hidden weights and biases
    w2_grad = np.dot(model['a1'],dz2.T)
    b2_grad = np.sum(dz2, axis=1)
    
    #local gradient delta
    dz1 = np.dot(model['w2'], dz2)* sigmoid_activation(model['a1'], derive = True)
    
    #gradient of first hidden weights and biases
    w1_grad = np.dot(model['a0'],dz1.T)
    b1_grad = np.sum(dz1, axis=1).reshape(num_hidden_unit2,-1)
    
    #embed delta
    dz0 = np.dot(model['w1'],dz1)


    #empty matrix for embed grads
    x_grad = np.zeros((250, num_hidden_unit1))

    #embed grad
    for i in range(3):
        x_matrix =  np.eye(250)[:,X[i,:]]
        dz0_channel = dz0[i * num_hidden_unit1 : (i+1) * num_hidden_unit1, :]
        x_grad = x_grad + np.dot(x_matrix,dz0_channel.T)

    
    #update deltas 
    model['dx'] = momentum_rate * model['dx'] + x_grad / batch_size
    model['dw1'] = momentum_rate * model['dw1'] + w1_grad / batch_size
    model['db1'] = momentum_rate * model['db1'] + b1_grad / batch_size
    model['dw2'] = momentum_rate * model['dw2'] + w2_grad / batch_size
    model['db2'] = momentum_rate * model['db2'] + b2_grad.reshape( model['db2'].shape[0],-1) / batch_size

    #update weights and biases
    model['embed']=  model['embed'] - lr_rate * model['dx']
    model['w1'] = model['w1'] - lr_rate * model['dw1']
    model['b1'] = model['b1'] - lr_rate * model['db1']
    model['w2'] = model['w2'] - lr_rate * model['dw2']
    model['b2'] = model['b2'] - lr_rate * model['db2']
    
def  predict(pred,y ):
    """
    This function predict accuracy
    
    Args:
        pred : predictions 
        y: labels
     

    Returns:
        accuracy
    """
    #true values
    c = np.argmax(pred,axis=0) - y 
    return len(c[c==0]) / c.shape[1]
    
def q1_forward(X, model):
    """
    This function predict accuracy
    
    Args:
        X : data 
        model : holds all data related to network
     

    Returns:
        activation functions results
    """


    num_hidden_unit1 = model['embed'].shape[1]
    

    #flat input
    X_flat = np.reshape(X, (1,-1), order="F").ravel()
  
    #flat embed
    embed_flat = model['embed'][X_flat].T
    
    #triagram embed values
    a0 = np.reshape(embed_flat, (num_hidden_unit1 * 3,-1), order="F")
    
    #forwards propagation
    z1 = np.dot(model['w1'].T,a0) + model['b1']
    a1 = sigmoid_activation(z1)
    z2 = np.dot(model['w2'].T,a1) + model['b2']
    pred = np.exp(z2 -np.amax(z2,axis=0))

   
    pred = pred / np.sum(pred,axis=0)

    return a0, a1, pred


def q1( lr_rate = 0.15, momentum_rate = 0.85, num_hidden_unit1 = 8, num_hidden_unit2 = 64, batch_size = 200, epochs = 50 ):
    """
    This function run q1
    
    """
   
    #read all data
    X_train, y_train, X_dev, y_dev, X_test, y_test, words = read_data(batch_size)

    #create model to hold everything related to networkd
    model = {}
    model['testx'] = X_test
    model['testy'] = y_test
    model['words'] = words

    num_batches = X_train.shape[2]

  
    # initialize weights and biases
    model['embed'] = normal_weight_and_bias_initializer((250,num_hidden_unit1), (num_hidden_unit1, 1))[0]
    model['w1'], model['b1'] = normal_weight_and_bias_initializer((3 * num_hidden_unit1, num_hidden_unit2), (num_hidden_unit2, 1))
    model['w2'], model['b2'] = normal_weight_and_bias_initializer((num_hidden_unit2, 250), (250, 1))

    #initialize deltas for momentum
    model['dx'] = np.zeros((250, num_hidden_unit1))
    model['dw1'] = np.zeros((3 * num_hidden_unit1, num_hidden_unit2))
    model['dw2'] = np.zeros((num_hidden_unit2, 250))
    model['db1'] = np.zeros((num_hidden_unit2, 1))
    model['db2'] = np.zeros((250, 1))


   
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        print("Epoch", epoch+1)
        train_batch_loss = []
        train_batch_acc = []
        
        #random shuffle get randomize result
        X_train, y_train = random_shuffle(X_train, y_train)
        for batch_no in range(num_batches):
            X_train_batch = X_train[:,:,batch_no]
            y_train_batch = y_train[:,:,batch_no]
            
            #forward propagation
            model['a0'], model['a1'], model['pred'] = q1_forward(X_train_batch, model )
            
            #create matrix with targets adjust size with predictions shape
            y_matrix =  np.eye(250)[:,y_train_batch.ravel()]
          
            #finding loss
            model['loss'] = cross_entropy(y_matrix, model['pred'])
           
            #hold loss
            train_batch_loss.append(model['loss'])
            train_batch_acc.append(predict(model['pred'], y_train_batch))
            #backpropagation
            backpropagation_q1(X_train_batch, y_matrix, momentum_rate, lr_rate, model)

        
        train_loss.append(np.mean(train_batch_loss))
        print("Train loss      :", np.mean(train_batch_loss), "------Train acc : ", np.mean(train_batch_acc))
        #find validation loss
        model['a0'], model['a1'],model['pred']= q1_forward(X_dev,model)


        y_dev_matrix =  np.eye(250)[:, y_dev.ravel()]
        
        model['loss'] = cross_entropy(y_dev_matrix, model['pred'] )
        val_loss.append(model['loss'])
        if( epoch == 0 ):
            print("Validation loss :", model['loss'],  '-----Validation accuracy : {}'.format(predict(model['pred'], y_dev)))
            model['val_loss'] = model['loss']
            
        elif( np.abs(model['val_loss'] - model['loss']) <= 0.0025 ):
            print("Validation loss :", model['loss'],  '-----Validation accuracy : {}'.format(predict(model['pred'], y_dev)))
            print('\nEarly stop due to insufficient reduction of validation loss\n')
            break
        else : 
            print("Validation loss :", model['loss'],  '-----Validation accuracy : {}'.format(predict(model['pred'], y_dev)))
            model['val_loss'] = model['loss']
       


    if epoch == (epochs - 1):
        print("\nTraining Completed\n")
  
  
    print("\nTest Results\n")

    model['a0'], model['a1'], model['pred'] = q1_forward(X_test, model)
    #find test loss
    y_test_matrix =  np.eye(250)[:, y_test.ravel()]
    model['loss'] = cross_entropy(y_test_matrix, model['pred'])
    print("Test loss : ", model['loss'] ,'-----Test accuracy : {}'.format(predict(model['pred'], y_test)))

    plt.plot(range(np.array(val_loss).shape[0]), np.array(val_loss)[:], label = 'Validation Loss')
    plt.plot(range(np.array(train_loss).shape[0]), np.array(train_loss)[:], label = 'Train Loss')
    plt.rcParams['figure.figsize'] = (10,10)
    plt.title('(D,P) = [{},{}]'.format(num_hidden_unit1, num_hidden_unit2))
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

   
    return model, np.array(train_loss), np.array(val_loss)


def plot_words(model):
    """
    This function apply TSNE and show results
    """
    
    print('Word Plot is Loading...')
    #TSNE transforms
    points = TSNE().fit_transform(model['embed'])
    texts = []
    plt.rcParams['font.weight'] = 'bold'
    #plt.rcParams['figure.figsize'] = (20, 30)
    for i in range(points.shape[0]):
        x = points[i,0]
        y = points[i,1]
        plt.plot(x, y, 'bo')
        plt.text(x * (1 + 0.01), y * (1 + 0.01) , str(model['words'].T[i][0]), fontsize=12)
        #texts.append(plt.text(x * (1 + 0.01), y * (1 + 0.01) , str(model['words'].T[i][0]), fontsize=17))
    #adjust_text(texts)
   
    #plt.savefig('textscatter.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

def close_words(model):
    
    #get same values
    np.random.seed(19)
    
    #select samples
    random_samples = np.random.randint(1,model['pred'].shape[1], size=(5))

    #get prediction values
    random_samplesx = model['pred'][:,random_samples].T
    #sort to find 10 max
    sorted_values = np.argsort(random_samplesx, axis=1)
    #find realted words
    words_found = sorted_values[:,-10:][:,::-1]
    
    #dimension reduction
    words_related = np.squeeze(model['words'][:,words_found])
    
    #tragrams
    triagrams = model['testx'][:,random_samples].T
    #triagram words
    triagram_words = np.squeeze(model['words'][:,triagrams])
    #one string triagram 
    triagram_words = np.array([ i[0] +" "+ i[1] +" "+ i[2] for i in triagram_words]).reshape(-1,1)
    #word table 
    word_table = np.hstack([triagram_words, words_related])
    #print
    for i in range(len(word_table)):
        string = str(word_table[i,0])+ ' : '
        for j in range(1, len(word_table[i,:])):
            string += str(word_table[i,j]) + '-->'
        print('\n'+string)
def q1_helper():    
    model, train_loss, val_loss = q1(epochs=50, num_hidden_unit1=16, num_hidden_unit2=128)
    plot_words(model)
    close_words(model)

    
question = sys.argv[1]

def sait_akturk_21501734_hw3(question):
    if question == '1' :
        print (question)
        q1_helper()
		
sait_akturk_21501734_hw3(question)