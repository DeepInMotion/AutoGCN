import json
import os
import shutil
import time
import torch


def save_checkpoint(model, optimizer, scheduler, epoch, epoch_time, actions, best_state, is_best, save_dir,
                    student_id, argmax=False):
    input_stream_state_dict = model.input_stream.state_dict(prefix="input_stream.")
    main_stream_state_dict = model.main_stream.state_dict(prefix="main_stream.")
    classifier_state_dict = model.classifier.state_dict(prefix="classifier.")

    combined_state_dict = {
        'input_stream': input_stream_state_dict,
        'main_stream': main_stream_state_dict,
        'classifier': classifier_state_dict
    }

    checkpoint = {
        'model': combined_state_dict, 'optimizer': optimizer, 'scheduler': scheduler,
        'best_state': best_state, 'actions': actions, 'epoch': epoch, 'epoch_time': epoch_time,
    }

    if argmax:
        student = 'argmax'
    else:
        student = 'student'

    save_student_dir = '{}/{}_{}'.format(save_dir, student, student_id)
    check_dir(save_student_dir)
    cp_name = '{}/checkpoint.pth.tar'.format(save_student_dir)
    torch.save(checkpoint, cp_name)
    if is_best:
        model_name = 'student_model_' + str(student_id)
        shutil.copy(cp_name, '{}/{}.pth.tar'.format(save_student_dir, model_name))
        with open('{}/reco_results_student_{}.json'.format(save_student_dir, student_id), 'w') as f:
            del best_state['cm']
            json.dump(best_state, f)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def get_current_timestamp():
    current_time = time.time()
    ms = int((current_time - int(current_time)) * 1000)
    return '[ {},{:0>3d} ] '.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time)), ms)


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)