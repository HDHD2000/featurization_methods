import numpy as np
import math
from persistence_methods import persistence_image_features
    
zero_dim_0 = []
one_dim_0 = []

def append_dim_list(dgms, dim_list):
    jth_pt = []
    for k in range(0, len(dgms)):
        if dgms[k][1] - dgms[k][0] >=0:
            birth = dgms[k][0]
            death = dgms[k][1]
        else:
            birth = dgms[k][1]
            death = dgms[k][0]
        if math.isinf(death):
            b = 50
        else:
            b = death
        t = [birth, b]
        jth_pt.append(t)
    dim_list.append(np.array(jth_pt))   
    
dgm = [np.array([[ 0.        ,  9.        ],
       [ 0.        , 10.        ],
       [ 0.        , 11.        ],
       [ 0.        , 12.        ],
       [ 0.        , 13.        ],
       [ 0.        , 14.        ],
       [ 0.        , 14.4222051 ],
       [ 0.        , 15.        ],
       [ 0.        , 19.13112647],
       [ 0.        , 26.2488095 ],
       [ 0.        , 41.35214626],
       [ 0.        , 54.68089246],
       [ 0.        ,         65]]), np.array([])]
    

append_dim_list(dgm[0], zero_dim_0)
append_dim_list(dgm[1], one_dim_0)

persistence_image_features(zero_dim_0, zero_dim_0)
persistence_image_features(one_dim_0, one_dim_0)