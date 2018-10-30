import numpy as np
from tqdm import tqdm

def get_data(pos_list, neg_list, N):
    data = {}
    np.random.shuffle(pos_list)
    np.random.shuffle(neg_list)
    
    data['positive_files'] = pos_list
    data['negative_files'] = neg_list

    data['X_positive'] = []
    data['X_negative'] = []

    for i in tqdm(range(N)):
        pos_fn = data['positive_files'][i]
        neg_fn = data['negative_files'][i]

        data['X_positive'].append(np.load(pos_fn))
        data['X_negative'].append(np.load(neg_fn))

    data['X_positive'] = np.array(data['X_positive'])
    data['Y_positive'] = np.ones((N))
    data['X_negative'] = np.array(data['X_negative'])
    data['Y_negative'] = np.zeros((N))

    data['X_'] = np.concatenate((data['X_positive'], data['X_negative']),axis=0)
    data['X'] = data['X_'].reshape((data['X_'].shape[0], int(data['X_'].shape[1]**2)))
    data['Y'] = np.concatenate((data['Y_positive'], data['Y_negative']),axis=0)

    return data
