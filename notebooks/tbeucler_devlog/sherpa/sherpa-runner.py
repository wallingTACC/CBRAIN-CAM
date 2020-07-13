import sherpa

def runner():
    parameters = [
                  sherpa.Discrete('num_units', [64, 512]),
                  sherpa.Discrete('num_layers', [2, 15]),
                  sherpa.Ordinal('batch_size', [4096]),
                  sherpa.Ordinal('learning_rate', [0.0001, 0.0003162278, 0.001, 0.003162278, 0.001]),
                  sherpa.Continuous('dropout_rate', [0, 0.6]),
                  sherpa.Continuous('leaky_relu_alpha', [0, 1]),  
                  sherpa.Ordinal('batch_norm', [0,1]),
                 ]
                  
    #alg = sherpa.algorithms.BayesianOptimization(max_num_trials=50)
    #alg = sherpa.algorithms.GPyOpt(max_num_trials=50)
    alg = sherpa.algorithms.RandomSearch(max_num_trials=40)

    #resources = ['/device:GPU:0', '/device:GPU:1', '/device:GPU:2', '/device:GPU:3']
    resources = [0, 1, 2, 3]
    
    scheduler = sherpa.schedulers.LocalScheduler(resources=resources)

    #study = sherpa.Study(parameters=parameters,
    #                     algorithm=alg,
    #                     lower_is_better=True)

    script = '/work/00157/walling/projects/cloud_emulator/walling-CBRAIN-CAM/notebooks/tbeucler_devlog/sherpa/sherpa-trial-phase2-study1-pos.py'
    tempdir = '/work/00157/walling/projects/cloud_emulator/walling-CBRAIN-CAM/notebooks/tbeucler_devlog/sherpa/sherpa-temp-phase2-study1-pos'

    #import time
    #time.sleep(1000)
    
    results = sherpa.optimize(parameters=parameters,
                              algorithm=alg,
                              lower_is_better=True,
                              filename=script,
                              output_dir=tempdir,
                              scheduler=scheduler,
                              max_concurrent=4,
                              verbose=1,
                              mongodb_args={'bind_ip':'0.0.0.0'}) #, 'port':47001})

if __name__=='__main__':
    runner()
