config={
        'experiment_id': "Stateful_LSTM",
        'data_folder': 'resources/data/discords/dutch_power/',
        'batch_size': 256,
        'n_epochs': 40,
        'dropout': 0.1 ,
        'look_back': 12,
        'look_ahead':1,
        'layers':{'input': 1, 'hidden1':20, 'hidden2':5,  'output': 1},
        'loss': 'mse',
        'train_test_ratio' : 0.7,
        'shuffle': False,
        'validation': True,
        'learning_rate': .02,
        'patience':20,
                           }
#
