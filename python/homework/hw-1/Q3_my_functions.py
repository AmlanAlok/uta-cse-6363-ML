import math


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

    ggg = gaussian_formula(
        result_store['M']['height_mean'],
        result_store['M']['height_variance'],
        test_input['input']['height']
    )


    print('height_m_estimate =', height_m_estimate)

    print('ending')
