import sherpa

def runner():
    parameters = [sherpa.Discrete('num_units', [64, 256]),
                  sherpa.Discrete('num_layers', [6, 24]),
                  sherpa.Ordinal('batch_size', [2048, 4096, 8192]),]
    #alg = sherpa.algorithms.BayesianOptimization(max_num_trials=50)

    alg = sherpa.algorithms.GPyOpt(max_num_trials=50)

    #resources = ['/device:GPU:0', '/device:GPU:1', '/device:GPU:2', '/device:GPU:3']
    resources = [0, 1, 2, 3]
    
    scheduler = sherpa.schedulers.LocalScheduler(resources=resources)

    #study = sherpa.Study(parameters=parameters,
    #                     algorithm=alg,
    #                     lower_is_better=True)

    script = '/home/dwalling/dev/sherpa/sherpa-trial.py'
    tempdir = '/home/dwalling/dev/sherpa/sherpa-temp'

    results = sherpa.optimize(parameters=parameters,
                              algorithm=alg,
                              lower_is_better=True,
                              filename=script,
                              output_dir=tempdir,
                              scheduler=scheduler,
                              max_concurrent=4,
                              verbose=1,
                              mongodb_args={'bind_ip':'localhost', 'port':27001})

if __name__=='__main__':
    runner()
