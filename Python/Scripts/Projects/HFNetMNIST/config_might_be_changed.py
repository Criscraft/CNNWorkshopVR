import os
from collections import OrderedDict

def get_config(gpu=0, seed=42):

    batch_size = 512
    epochs = 100
    lr_init = batch_size / 64. * 1. * 1e-3
    lr_decay_start = 50
    n_tests = 20
    norm_mean = 0.13
    norm_std = 0.31
    pytorchtools_path = '/nfshome/linse/Documents/development/pytorch_classification/pytorchtools'
    script_dir_path = '/nfshome/linse/Documents/sandbox/CNN_node_editor/training_mnist/scripts'

    config = {
        'log_path' : f'/nfshome/linse/Documents/sandbox/CNN_node_editor/training_mnist/log/mnist_hfnet_blocks_2_2_2_2_shallow',
        'info' : "",
        'seed' : seed,
        'epochs' : epochs,
        'num_workers' : 2,
        'pin_memory' : False,
        'no_cuda' : False,
        'cuda_device' : f'cuda:{gpu}',
        'save_data_paths' : True,
        'path_of_configfile' : os.path.abspath(__file__),
    }

    
    #networks
    config['networks'] = {}

    item = {}; config['networks']['network_main'] = item
    item['source'] = os.path.join(script_dir_path, 'HFNetNoStride2LayerNormGroupConv1x1ConvOutput.py')
    item['params'] = {
        'n_classes' : 10,
        'start_config' : {
            'k' : 3, 
            'filter_mode' : 'UnevenPosOnly',
            'n_angles' : 4,
            'n_channels_in' : 1,
            'n_channels_out' : 20, # muss teilbar durch shuffle_conv_groups sein sowie durch die Anzahl an Klassen 
            'f' : 4,
            'handcrafted_filters_require_grad' : False,
            'shuffle_conv_groups' : 1,
        },
        'blockconfig_list' : [
            {'k' : 3, 
            'filter_mode_1' : 'UnevenPosOnly',
            'filter_mode_2' : 'UnevenPosOnly',
            'n_angles' : 4,
            'n_blocks' : 2,
            'n_channels_in' : 20,
            'n_channels_out' : 20,
            'avgpool' : True if i>0 else False,
            'f' : 1,
            'handcrafted_filters_require_grad' : False,
            'shuffle_conv_groups' : 20 // 4,
            } for i in range(4)],
        'avgpool_after_firstlayer' : False,
    }
    

    #loss functions
    config['loss_fns'] = {}

    item = {}; config['loss_fns']['loss_fn_main'] = item
    item['source'] = os.path.join(pytorchtools_path, 'ptutils/CrossEntropyLoss.py')
    item['params'] = {}


    #optimizers
    config['optimizers'] = {}

    item = {}; config['optimizers']['optimizer_main'] = item
    item['source'] = os.path.join(pytorchtools_path, 'ptutils/Lamb2.py')
    item['params'] = {
        'lr' : lr_init, 
        'betas' : (0.9, 0.999),
    }


    #transformations
    config['transforms'] = {}

    item = {}; config['transforms']['transform_train'] = item
    item['source'] = os.path.join(script_dir_path, 'TransformTestGS.py')
    item['params'] = {
        'norm_mean' :  norm_mean, 
        'norm_std' : norm_std,
        }
    
    item = {}; config['transforms']['transform_test'] = item
    item['source'] = os.path.join(script_dir_path, 'TransformTestGS.py')
    item['params'] = {
        'norm_mean' :  norm_mean, 
        'norm_std' : norm_std,
        }


    #datasets
    config['datasets'] = {}

    item = {}; config['datasets']['dataset_train'] = item
    item['source'] = os.path.join(script_dir_path, 'MNIST.py')
    item['params'] = {
        'b_train' : True,
        'transform' : 'transform_train',
        'root' : '/data_ssd0/linse/tmp_data',
        'download' : True,
    }

    item = {}; config['datasets']['dataset_test'] = item
    item['source'] = os.path.join(script_dir_path, 'MNIST.py')
    item['params'] = {
        'b_train' : False,
        'transform' : 'transform_test',
        'root' : '/data_ssd0/linse/tmp_data',
        'download' : False,
    }


    #loaders
    config['loaders'] = {}

    item = {}; config['loaders']['loader_train'] = item
    item['source'] = os.path.join(pytorchtools_path, 'ptutils/DataloaderGetter.py')
    item['params'] = {
        'dataset' : 'dataset_train',
        'batch_size' : batch_size,
        'b_shuffled' : True,
        'custom_seed' : 42,
    }

    item = {}; config['loaders']['loader_test'] = item
    item['source'] = os.path.join(pytorchtools_path, 'ptutils/DataloaderGetter.py')
    item['params'] = {
        'dataset' : 'dataset_test',
        'batch_size' : batch_size,
        'b_shuffled' : False,
        'custom_seed' : 42,
    }


    #scheduler modules
    config['scheduler_modules'] = OrderedDict()

    item = {}; config['scheduler_modules']['training_initializer'] = item
    item['source'] = os.path.join(pytorchtools_path, 'ptschedulers/SchedulerTrainingInitializationModule.py')
    item['params'] = {}


    item = {}; config['scheduler_modules']['scheduler_trainer'] = item
    item['source'] = os.path.join(pytorchtools_path, 'ptschedulers/SchedulerTrainingModule.py')
    item['params'] = {'clamp_weights_range' : [-1., 1.]}

    
    item = {}; config['scheduler_modules']['scheduler_validator'] = item
    item['source'] = os.path.join(pytorchtools_path, 'ptschedulers/SchedulerValidationModule.py')
    item['params'] = {
        'active_epochs' : [round(i * config['epochs'] / n_tests) for i in range(1, n_tests)],
    }

    item = {}; config['scheduler_modules']['scheduler_model_saver'] = item
    item['source'] = os.path.join(pytorchtools_path, 'ptschedulers/SchedulerSaveNetModule.py')
    item['params'] = {}


    item = {}; config['scheduler_modules']['scheduler_lr'] = item
    item['source'] = os.path.join(pytorchtools_path, 'ptschedulers/SchedulerLearningRateModifierModule.py')
    item['params'] = {
        'schedule' : {
            lr_decay_start : lr_init * 3e-1, 
            lr_decay_start + int(5. / 10. * (config['epochs'] - lr_decay_start)) : lr_init * 1e-1, 
            lr_decay_start + int(6. / 10. * (config['epochs'] - lr_decay_start)) : lr_init * 3e-2, 
            lr_decay_start + int(7. / 10. * (config['epochs'] - lr_decay_start)) : lr_init * 1e-2, 
            lr_decay_start + int(8. / 10. * (config['epochs'] - lr_decay_start)) : lr_init * 3e-3, 
            lr_decay_start + int(9. / 10. * (config['epochs'] - lr_decay_start)) : lr_init * 1e-3,
        }
    }

    return config
    