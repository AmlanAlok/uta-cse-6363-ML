import math


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


def gaussian_formula(mean, var, x):
    pie = math.pi
    e = 2.71828
    r = (1/math.sqrt(2*pie*var)) * pow(e, (-1*pow(x-mean, 2))/(2*var))
    print('gaussian output =', r)
    return r


def gaussian_naive_bayes(input_data, test_input):

    result_store = {}
    w_count = 0
    m_count = 0
    height_w_total = 0
    height_m_total = 0
    weight_w_total = 0
    weight_m_total = 0
    age_w_total, age_m_total = 0, 0

    for dp in input_data:

        if dp['output'] == 'M':
            m_count += 1
            height_m_total += dp['input']['height']
            weight_m_total += dp['input']['weight']
            age_m_total += dp['input']['age']
        elif dp['output'] == 'W':
            w_count += 1
            height_w_total += dp['input']['height']
            weight_w_total += dp['input']['weight']
            age_w_total += dp['input']['age']

    result_store['M'] = {
        'count': m_count,
        'height_total': height_m_total,
        'weight_total': weight_m_total,
        'age_total': age_m_total,
        'height_mean': height_m_total/ m_count,
        'weight_mean': weight_m_total/ m_count,
        'age_mean': age_m_total/ m_count
    }

    result_store['W'] = {
        'count': w_count,
        'height_total': height_w_total,
        'weight_total': weight_w_total,
        'age_total': age_w_total,
        'height_mean': height_w_total / w_count,
        'weight_mean': weight_w_total / w_count,
        'age_mean': age_w_total / w_count
    }

    height_m_diff_mean_square_total, height_w_diff_mean_square_total = 0, 0

    for dp in input_data:

        if dp['output'] == 'M':
            height_m_diff_mean_square_total += pow(dp['input']['height'] - result_store[dp['output']]['height_mean'], 2)
        elif dp['output'] == 'W':
            height_w_diff_mean_square_total += pow(dp['input']['height'] - result_store[dp['output']]['height_mean'], 2)

    result_store['M']['height_variance'] = height_m_diff_mean_square_total/ (result_store['M']['count']-1)
    result_store['W']['height_variance'] = height_w_diff_mean_square_total / (result_store['W']['count']-1)

    result_store['M']['probability'] = result_store['M']['count'] / (result_store['M']['count'] + result_store['W']['count'])
    result_store['W']['probability'] = result_store['W']['count'] / (result_store['M']['count'] + result_store['W']['count'])

    height_m_estimate = result_store['M']['probability'] * gaussian_formula(
        result_store['M']['height_mean'],
        result_store['M']['height_variance'],
        test_input['input']['height']
    )

    print('height_m_estimate =', height_m_estimate)

    print('ending')


def gaussian_naive_bayes_2(input_data, test_input):

    input_dict = {}

    for dp in input_data:
        if dp[OUTPUT] in input_dict:
            input_dict[dp[OUTPUT]][COUNT] += 1
        else:
            input_dict[dp[OUTPUT]] = {
                COUNT: 1,
                HEIGHT_TOTAL: 0, HEIGHT_MEAN: 0, HEIGHT_VAR: 0, HEIGHT_MEAN_SQR_TOTAL: 0,
                WEIGHT_TOTAL: 0, WEIGHT_MEAN: 0, WEIGHT_VAR: 0, WEIGHT_MEAN_SQR_TOTAL: 0,
                AGE_TOTAL: 0, AGE_MEAN: 0, AGE_VAR: 0, AGE_MEAN_SQR_TOTAL: 0,
            }

    output_labels = input_dict.keys()
    a = type(output_labels)

    ''' totaling all features '''
    for dp in input_data:
        input_dict[dp[OUTPUT]][HEIGHT_TOTAL] += dp[INPUT][HEIGHT]
        input_dict[dp[OUTPUT]][WEIGHT_TOTAL] += dp[INPUT][WEIGHT]
        input_dict[dp[OUTPUT]][AGE_TOTAL] += dp[INPUT][AGE]

    ''' Calculating Mean '''
    for label in output_labels:
        input_dict[label][HEIGHT_MEAN] = input_dict[label][HEIGHT_TOTAL]/ input_dict[label][COUNT]
        input_dict[label][WEIGHT_MEAN] = input_dict[label][WEIGHT_TOTAL] / input_dict[label][COUNT]
        input_dict[label][AGE_MEAN] = input_dict[label][AGE_TOTAL]/ input_dict[label][COUNT]

    ''' Calculating SQR of difference from Mean for each data point'''
    for dp in input_data:
        input_dict[dp[OUTPUT]][HEIGHT_MEAN_SQR_TOTAL] += pow(dp[INPUT][HEIGHT] - input_dict[dp[OUTPUT]][HEIGHT_MEAN], 2)
        input_dict[dp[OUTPUT]][WEIGHT_MEAN_SQR_TOTAL] += pow(dp[INPUT][WEIGHT] - input_dict[dp[OUTPUT]][WEIGHT_MEAN], 2)
        input_dict[dp[OUTPUT]][AGE_MEAN_SQR_TOTAL] += pow(dp[INPUT][AGE] - input_dict[dp[OUTPUT]][AGE_MEAN], 2)

    ''' Calculating Variance '''
    for label in output_labels:
        input_dict[label][HEIGHT_VAR] = input_dict[label][HEIGHT_MEAN_SQR_TOTAL] / (input_dict[label][COUNT]-1)
        input_dict[label][WEIGHT_VAR] = input_dict[label][WEIGHT_MEAN_SQR_TOTAL] / (input_dict[label][COUNT] - 1)
        input_dict[label][AGE_VAR] = input_dict[label][AGE_MEAN_SQR_TOTAL] / (input_dict[label][COUNT] - 1)




    print('ending')