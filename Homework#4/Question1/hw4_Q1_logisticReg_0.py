# The following web address guided me to write my code (March 2022): https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2
import numpy as np
import matplotlib.pyplot as plt

def logistic_test(data, weigths):
    # calculate predictions
    preds=sigmoid(np.dot(data,weigths))
    # create list to store predictions
    pred_class=[]

    # if preds_i >= 0.5 --> assign value of 1 / preds_i is a single prediction value in the array preds
    # if preds_i < 0.5 --> assign value of 0
    pred_class = [1 if i >= 0.5 else 0 for i in preds]
    return np.array(pred_class)


def gradient(data, labels, labels_pred):
    labels=np.array([labels]).T
    # m is the number of training samples
    m=data.shape[0]
    # Gradient of loss w.r.t. weights . This derivation is for encoding lables as C+=+1 and C-=0 from lecture notes
    dw=(1/m)*np.dot(data.T,(labels_pred-labels))
    return dw


def sigmoid(s):
    return 1/(1 + np.exp(-s))


def logistic_train(data, labels, epsilon=1e-5, maxiter=1000, step_size=0.00001):
    """
        code to train a logistic regression classifier
        %
        % INPUTS:
        % data = n * (d+1) matrix withn samples and d features, where
        %       column d+1 is all ones (corresponding to the intercept term)
        % labels = n * 1 vector of class labels (taking values 0 or 1)
        % epsilon = optional argument specifying the convergence
        %           criterion - if the change in the absolute difference in
        %           predictions, from one iteration to the next, averaged across
        %           input features, is less than epsilon, then halt
        %           (if unspecified, use a default value of 1e-5)
        % maxiter = optional argument that specifies the maximum number of
        %           iterations to execute (useful when debugging in case your
        %           code is not converging correctly!)
        %           (if unspecified can be set to 1000)
        %
        % OUTPUT:
        %   weights = (d+1) * 1 vector of weights where the weights correspond to
        %               the columns of "data"

        """
    feature_dim = data.shape[1]
    feature_dim_n = data.shape[0]
    # initialize weights & labels_pred
    weight = np.zeros((feature_dim, 1))
    labels_pred = np.zeros((feature_dim_n, 1))

    # create list of obj values
    objective_value = []
    for i in range(maxiter):
        print(i)
        # initialize labels_pred_old
        labels_pred_old=labels_pred.copy()
        # calculating prediction
        labels_pred = sigmoid(np.dot(data,weight))
        # Calculate the gradient of loss w.r.t weight
        grad=gradient(data, labels, labels_pred)
        # update weight
        weight -= step_size * grad
        # stop criteria
        objval_i=np.mean(np.absolute(labels_pred - labels_pred_old))
        objective_value.append(objval_i)
        if  objval_i< epsilon:
            print("stop criterion met")
            break

    return weight, objective_value


def exp():
    # load dataset
    x_data=np.loadtxt("C:/Users/quispeab/Desktop/ml_hw4/spam_email/data.txt")#
    y_label=np.loadtxt("C:/Users/quispeab/Desktop/ml_hw4/spam_email/labels.txt")
    
    # subsample
    n=2000 # <========= INPUT ( 200; 500; 800, 1000; 1500, 2000)
    # train dataset
    feature_train=x_data[:n,:]
    # add column
    dim_add_col=feature_train.shape[0]
    add_col=np.ones([dim_add_col,1])
    feature_train=np.concatenate((feature_train,add_col),1)
    # train target
    target_train=y_label[:n,]

    # test dataset
    feature_test=x_data[2000:4602,:]
    # add column
    dim_add_col=feature_test.shape[0]
    add_col=np.ones([dim_add_col,1])
    feature_test=np.concatenate((feature_test,add_col),1)
    # test target
    target_test=y_label[2000:4602,]

    # training the model
    maxiter=50000
    epsilon=1e-6 #1e-5
    step_size=0.01 # 0.01, 0.00015
    weights, obj_val=logistic_train(feature_train, target_train, epsilon, maxiter, step_size)
    
    # Plot to check "convergence criterion"
    obj_val=obj_val[1:]
    # plot
    fig=plt.figure()
    plt.plot(range(len(obj_val)), obj_val, linestyle='-', color='r', label='Objective Value')
    plt.xlabel("Iteration")
    plt.ylabel("Objective_value")
    plt.title("Convergence criterion")
    plt.show()
    print("training is complete")

    # evaluate accuracy on the test dataset
    labels_predict_test=logistic_test(feature_test, weights)

    accuracy = np.sum(target_test == labels_predict_test) / len(target_test)
    print("the accuracy based on {} train samples".format(n))
    print("The accuracy of the model is:", accuracy)


if __name__ == '__main__':

    # experiment: training & testing.
    exp()
