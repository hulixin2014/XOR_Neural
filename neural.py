# coding=utf-8
import numpy as np


# TODO: THis is a packet the build a two layers neural to deal with the
# TODO: problem of XOR 's linear separable.


def active_fun(input_x):
    """
    Activation function that sort the data!

    Args:
        input_x (float):
            the input of function!

    Return:
        number of integer:
            the result of sorting the data!
    """
    # use sigmoil function
    result = 1.0 / (1 + np.exp(-input_x))
    if result >= 0.5:
        return 1
    else:
        return 0


def active_derivative(input_x):
    """
    Calculate the sigmoil's derivative!

    Args:
        input_x (float):
            the input of function!

    Return:
        number of float :
            the result of evaluation of functions!
    """
    result = 1.0 / (1 + np.exp(-input_x))
    return result * (1 - result)


def predict(array_weight1, array_weight2, array_weight3, array_input):
    """
        Predict the result!weight=[b1,w1,w2],input=[1,x1,x2]

        Args:
            array_weight1 (array of float):
                the weights of the first node of the first layer of neural!
            array_weight2 (array of float):
                the weights of the second node of the first layer of neural!
            array_weight3 (array of float):
                the weights of the first node of the second layer of neural!
            array_input (array of float):
                the input's data of predicting!
        Return:
            number of integer:
            all the dots' results of predicting!
        """
    array_weight1 = np.array(array_weight1)
    array_weight2 = np.array(array_weight2)
    array_weight3 = np.array(array_weight3)
    array_input = np.array(array_input)
    # calculate the first layer
    # print array_weight1,array_input
    result1 = active_fun(sum(array_weight1 * array_input))
    result2 = active_fun(sum(array_weight2 * array_input))
    # calculate the second layer
    array_two = np.array([1, result1, result2])
    result3 = active_fun(sum(array_weight3 * array_two))
    return result1, result2, result3


def train(array_weight1, array_weight2, array_weight3, array_input, array_result, learn_rate, e=0):
    """
    Train the neural!weight=[b1,w1,w2],input=[[1,x1,x2],[1,x11,x22]].
    The first weight points to the first dot,and so on.

    Args:
        array_weight1 (array of float):
            the weights of the first node of the first layer of neural!
        array_weight2 (array of float):
            the weights of the second node of the first layer of neural!
        array_weight3 (array of float):
            the weights of the first node of the second layer of neural!
        array_input (array of float):
            the input's data of training neural!
        array_result (array of float):
            the output's expected data of training neural!
        learn_rate (float):
            a float number of learning rate
        e (float):
             Error rate.
    Return:
        the array of weights:
            the result of  having corrected the  initial weights!
    """
    array_weight1 = np.array(array_weight1)
    array_weight2 = np.array(array_weight2)
    array_weight3 = np.array(array_weight3)
    array_input = np.array(array_input)
    real_e = 1
    i = 1
    length = len(array_result)
    while real_e > e:
        real_e = 0
        print "第%d轮权值修正：" % (i)
        for idx in range(length):
            result1, result2, result3 = predict(array_weight1, array_weight2, array_weight3, array_input[idx])
            if result3 != array_result[idx]:
                print "x1={0}和x2={1}时，预测值={2}，期望值={3}".format(
                    array_input[idx][1], array_input[idx][2], result3, array_result[idx])
                # difference value back propagation
                diff_v3 = array_result[idx] - result3
                diff_v2 = diff_v3 * array_weight3[2]
                diff_v1 = diff_v3 * array_weight3[1]
                print "反向传播差值a3={0}，a2={1}，a1={2}".format(diff_v3, diff_v2, diff_v1)
                # correct the weights
                sum_value = sum(array_weight1 * array_input[idx])
                array_weight1 = array_weight1 + learn_rate * diff_v1 * array_input[idx] * active_derivative(sum_value)
                sum_value = sum(array_weight2 * array_input[idx])
                array_weight2 = array_weight2 + learn_rate * diff_v2 * array_input[idx] * active_derivative(sum_value)
                array_two = np.array([1, result1, result2])
                sum_value = sum(array_weight3 * array_two)
                array_weight3 = array_weight3 + learn_rate * diff_v3 * array_two * active_derivative(sum_value)
                print array_weight1, array_weight2, array_weight3
                real_e += 1
        i += 1
        # calculate error rate
        real_e = float(real_e) / length
    return array_weight1, array_weight2, array_weight3
