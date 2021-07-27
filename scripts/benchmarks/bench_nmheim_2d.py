import pytest
import joblib
import numpy as np

from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp

from esn_nmheim.input_map import InputMap
from esn_nmheim.utils import split_train_label_pred
from esn_nmheim.toydata import gauss2d_sequence, mackey2d_sequence
from esn_nmheim.imed import imed
import esn_nmheim.sparse_esn as se

from time import time
import jax

def sparse_esn_2d_train_pred(tmpdir, data, specs,
                             spectral_radius=1.5,
                             density=0.01,
                             Ntrans=500,        # Number of transient initial steps before training
                             Ntrain=2500,       # Number of steps to train on
                             Npred=500,         # Number of steps for free-running prediction
                             mse_threshold=1e-5,
                             plot_prediction=False):
    start = time()
    #np.random.seed(1)
    N = Ntrain + Npred + 1
    assert data.ndim == 3
    assert data.shape[0] >= N

    # prepare data
    from esn_dev.utils import  split_train_label_pred 
    #inputs, labels, pred_labels = split_train_label_pred(data,Ntrain,Npred)
    inputs, labels, pred_labels  = split_train_label_pred(data, Ntrain, Npred,Ntrans)
    img_shape = inputs.shape[1:]
    

    
    # build esn
    map_ih = InputMap(specs)
    hidden_size = map_ih.output_size(img_shape)
    print("Hidden size: ", hidden_size)
    
    density = density/hidden_size # for approx fixed nz per row 
    
    #to save time and stuff
    bench_dic = dict(input_dim = img_shape[0]*img_shape[1],
                    Nhidden = hidden_size)

    build_esn_start = time() 
    esn = se.esncell(map_ih, hidden_size, spectral_radius=spectral_radius, density=density)
    esn[-1].block_until_ready()
    build_esn_end = time()
    bench_dic['build_esn'] = build_esn_end-build_esn_start
    
    print(f"finished building after {time()-start}s")
    # compute training states
    train_start = time()
    H = se.augmented_state_matrix(esn, inputs, Ntrans)
    H.block_until_ready()
    train_end   = time()
    
    bench_dic['train'] = train_end-train_start
    
    print(f"Finished harvesting after {time()-start}s")
    # compute last layer without imed

    imed_lstsq_start = time()
    labels = labels.reshape(inputs.shape[0]-Ntrans, -1)
    #model = se.train(esn, H, _labels[Ntrans:])
    # and with imed
    model = se.train_imed(esn, H, inputs[Ntrans:], labels, sigma=1.)
    model[-1].block_until_ready()
    imed_lstsq_end = time()

    bench_dic['imed_lstsq'] = imed_lstsq_end-imed_lstsq_start
    bench_dic['lstsq'] = 0 # only for new esn
    
    print(f"Finished training after {time()-start}s")
    # predict
    y0, h0 = labels[-1], H[-1]
    predict_start = time()
    (y,h), (ys,hs) = se.predict(model, y0.reshape(img_shape), h0, Npred)
    y.block_until_ready()
    predict_end = time()
    bench_dic['predict'] = predict_end-predict_start

    
    print(f"Finished predicting after {time()-start}s")
    # predict with warump of Ntrain frames
    #_, (wys,_) = se.warmup_predict(model, labels[-Ntrans:], Npred)

    """if plot_prediction:
        import matplotlib.pyplot as plt
        from AnomalyDetectionESN.visualize import animate_double_imshow
        anim = animate_double_imshow(ys, pred_labels)

        fig, ax = plt.subplots(1,2)
        ax[0].imshow(pred_labels.sum(axis=0))
        ax[0].set_title("Truth")
        ax[1].imshow(ys.sum(axis=0))
        ax[1].set_title("Pred.")
        plt.show()"""

    mse = jnp.mean((ys - pred_labels)**2)
    bench_dic['mse'] = float(mse)
    bench_dic['imed']  = 0

    #w_mse = jnp.mean((wys[25] - pred_labels[25])**2)
    print("MSE:  ", mse)
    """# print("IMED: ", imed(ys, pred_labels)[25])
    assert mse < mse_threshold
    assert w_mse < mse_threshold
    assert jnp.isclose(mse, w_mse, atol=1e-3)

    with open(tmpdir / "esn.pkl", "wb") as fi:
        joblib.dump(model, fi)
    pkl_model = se.load_model(tmpdir / "esn.pkl")
    _, (pkl_ys,_) = se.predict(pkl_model, y0, h0, Npred)
    print(ys[0])
    print(pkl_ys)
    assert jnp.all(jnp.isclose(pkl_ys, ys))"""
    
    end = time()
    print(f'To run all, took {start-end:.1f}s')
    return bench_dic

def test_sparse_esn_lissajous(tmpdir,input_dim,data=None):
    input_shape = (input_dim,input_dim)
    input_size  = input_shape[0] * input_shape[1]

    if data is None:

        from esn.utils import scale
        gauss2d_sequence(N=600+100+100+1,size=(size,size),dtype='float64')
        data = scale(data, -1, 1)

    specs = [
        {"type": "pixels", "size": input_shape, "factor": 1.},
        {"type": "dct", "size": [15, 15], "factor": 1.},
        {"type": "gradient", "factor": 4.},
        {"type":"conv", "size":(3,3),   "kernel":"random",  "factor": 1.},
        {"type":"random_weights", "input_size":input_size, "hidden_size":3500, "factor": 1.},
    ]
    
    Ntrain = 600
    Ntrans = 100
    Npred  = 100
    
    nzper_row = 10
    
    bench_dic = sparse_esn_2d_train_pred(tmpdir, data, specs,
        plot_prediction=False, mse_threshold=1e-15,
        spectral_radius=2.0, density=nzper_row,
        Ntrain=Ntrain, Npred=Npred, Ntrans=Ntrans)
    import pandas as pd
    import numpy as np
    import os
    file = 'nmheim_benchmark.csv' #/home/jfelding/esn_benchmark/
    #print(os.getcwd())
    #print(bench_dic.keys())
    df = pd.read_csv(file)
    df = df.append(bench_dic, ignore_index=True)
    #print(df)
    df.to_csv(file, index=False,) 

import numpy as np
    
if __name__ == "__main__":
    from esn_dev.toydata import gauss2d_sequence
    #test_sparse_esn_lissajous("tmp")
    #jax.profiler.start_trace("tmp")
    preds_per_problem = 5
    sizes = np.arange(start=120,stop=505,step=10)
    #sizes = [60]
    for size in sizes:
        for i in range(preds_per_problem):
            #test_sparse_esn_lissajous("tmp",input_dim=size)
            from esn.utils import scale
            data = gauss2d_sequence(N=600+100+100+1,size=(size,size),dtype='float64')
            data = scale(data, -1, 1)

            while True:
                try:
                    test_sparse_esn_lissajous("tmp",input_dim=size,data=data)
                except Exception as e:
                    print(e)
                    print('Trying again!')
                    continue #just try again! 
                break
