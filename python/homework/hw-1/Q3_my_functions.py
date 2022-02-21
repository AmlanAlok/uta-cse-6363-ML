import math
import inspect

OUTPUT = 'output'
INPUT = 'input'
COUNT = 'count'
HEIGHT = 'height'
WEIGHT = 'weight'
AGE = 'age'
HEIGHT_TOTAL = 'height_total'
HEIGHT_MEAN = 'height_mean'
HEIGHT_VAR = 'height_var'
HEIGHT_MEAN_SQR_TOTAL = 'height_mean_sqr_total'
WEIGHT_TOTAL = 'weight_total'
WEIGHT_MEAN = 'weight_mean'
WEIGHT_VAR = 'weight_var'
WEIGHT_MEAN_SQR_TOTAL = 'weight_mean_sqr_total'
AGE_TOTAL = 'age_total'
AGE_MEAN = 'age_mean'
AGE_VAR = 'age_var'
AGE_MEAN_SQR_TOTAL = 'age_mean_sqr_total'
CLASS_PROBABILITY = 'probability'


def gaussian_formula(mean, var, test_parameter):
    pi = math.pi
    e = math.e
    r = (1/math.sqrt(2*pi*var)) * pow(e, (-1*pow(test_parameter-mean, 2))/(2*var))
    # print('gaussian output =', r)
    return r


def leave_one_out(input_data, exclude_age=False):
    print('inside func ' + inspect.stack()[0][3])

    result_dict = {
        'accuracy': '',
        'result': [None] * len(input_data)
    }

    for left_out_dp in input_data:

        input_dict = {}

        for dp in input_data:

            if dp['index'] != left_out_dp['index']:
                if dp[OUTPUT] in input_dict:
                    input_dict[dp[OUTPUT]][COUNT] += 1
                else:
                    input_dict[dp[OUTPUT]] = {
                        COUNT: 1, CLASS_PROBABILITY: 0,
                        HEIGHT_TOTAL: 0, HEIGHT_MEAN: 0, HEIGHT_VAR: 0, HEIGHT_MEAN_SQR_TOTAL: 0,
                        WEIGHT_TOTAL: 0, WEIGHT_MEAN: 0, WEIGHT_VAR: 0, WEIGHT_MEAN_SQR_TOTAL: 0,
                        AGE_TOTAL: 0, AGE_MEAN: 0, AGE_VAR: 0, AGE_MEAN_SQR_TOTAL: 0,
                    }

        test_input_record = {
            'input': {
                'height': float(left_out_dp[INPUT][HEIGHT]),
                'weight': float(left_out_dp[INPUT][WEIGHT]),
                'age': int(left_out_dp[INPUT][AGE])
            },
            'output': left_out_dp[OUTPUT]
        }

        test_input_record = get_prediction(input_data, input_dict, test_input_record)

        if test_input_record[OUTPUT] == test_input_record['prediction']:
            result_dict['result'][left_out_dp['index']] = 1
        else:
            result_dict['result'][left_out_dp['index']] = 0

    result_dict['accuracy'] = (sum(result_dict['result'])/ len(result_dict['result'])) * 100


    print('END')



def gaussian_naive_bayes(input_data, test_input):

    input_dict = {}

    for dp in input_data:
        if dp[OUTPUT] in input_dict:
            input_dict[dp[OUTPUT]][COUNT] += 1
        else:
            input_dict[dp[OUTPUT]] = {
                COUNT: 1, CLASS_PROBABILITY: 0,
                HEIGHT_TOTAL: 0, HEIGHT_MEAN: 0, HEIGHT_VAR: 0, HEIGHT_MEAN_SQR_TOTAL: 0,
                WEIGHT_TOTAL: 0, WEIGHT_MEAN: 0, WEIGHT_VAR: 0, WEIGHT_MEAN_SQR_TOTAL: 0,
                AGE_TOTAL: 0, AGE_MEAN: 0, AGE_VAR: 0, AGE_MEAN_SQR_TOTAL: 0,
            }

    return get_prediction(input_data, input_dict, test_input)


def get_prediction(input_data, input_dict, test_input):

    output_labels = input_dict.keys()

    ''' totaling all features '''
    for dp in input_data:
        input_dict[dp[OUTPUT]][HEIGHT_TOTAL] += dp[INPUT][HEIGHT]
        input_dict[dp[OUTPUT]][WEIGHT_TOTAL] += dp[INPUT][WEIGHT]
        input_dict[dp[OUTPUT]][AGE_TOTAL] += dp[INPUT][AGE]

    ''' Calculating Mean and Class Probability'''
    for label in output_labels:
        input_dict[label][HEIGHT_MEAN] = input_dict[label][HEIGHT_TOTAL] / input_dict[label][COUNT]
        input_dict[label][WEIGHT_MEAN] = input_dict[label][WEIGHT_TOTAL] / input_dict[label][COUNT]
        input_dict[label][AGE_MEAN] = input_dict[label][AGE_TOTAL] / input_dict[label][COUNT]
        input_dict[label][CLASS_PROBABILITY] = input_dict[label][COUNT] / len(input_data)

    ''' Calculating SQR of difference from Mean for each data point'''
    for dp in input_data:
        input_dict[dp[OUTPUT]][HEIGHT_MEAN_SQR_TOTAL] += pow(dp[INPUT][HEIGHT] - input_dict[dp[OUTPUT]][HEIGHT_MEAN], 2)
        input_dict[dp[OUTPUT]][WEIGHT_MEAN_SQR_TOTAL] += pow(dp[INPUT][WEIGHT] - input_dict[dp[OUTPUT]][WEIGHT_MEAN], 2)
        input_dict[dp[OUTPUT]][AGE_MEAN_SQR_TOTAL] += pow(dp[INPUT][AGE] - input_dict[dp[OUTPUT]][AGE_MEAN], 2)

    ''' Calculating Variance '''
    for label in output_labels:
        input_dict[label][HEIGHT_VAR] = input_dict[label][HEIGHT_MEAN_SQR_TOTAL] / (input_dict[label][COUNT] - 1)
        input_dict[label][WEIGHT_VAR] = input_dict[label][WEIGHT_MEAN_SQR_TOTAL] / (input_dict[label][COUNT] - 1)
        input_dict[label][AGE_VAR] = input_dict[label][AGE_MEAN_SQR_TOTAL] / (input_dict[label][COUNT] - 1)

    test_input['probability'] = {}
    max_probability = 0.0

    for label in output_labels:
        test_input['probability'][label] = {
            'P(height|C)': gaussian_formula(mean=input_dict[label][HEIGHT_MEAN], var=input_dict[label][HEIGHT_VAR],
                                            test_parameter=test_input[INPUT][HEIGHT]),
            'P(weight|C)': gaussian_formula(mean=input_dict[label][WEIGHT_MEAN], var=input_dict[label][WEIGHT_VAR],
                                            test_parameter=test_input[INPUT][WEIGHT]),
            'P(age|C)': gaussian_formula(mean=input_dict[label][AGE_MEAN], var=input_dict[label][AGE_VAR],
                                         test_parameter=test_input[INPUT][AGE])
        }

        test_input['probability'][label]['final_estimate'] = input_dict[label][CLASS_PROBABILITY] * \
                                                             test_input['probability'][label]['P(height|C)'] * \
                                                             test_input['probability'][label]['P(weight|C)'] * \
                                                             test_input['probability'][label]['P(age|C)']

        print('For', label, 'Final Estimate =', test_input['probability'][label]['final_estimate'])

        if test_input['probability'][label]['final_estimate'] > max_probability:
            max_probability = test_input['probability'][label]['final_estimate']
            test_input['prediction'] = label

    return test_input
