import os
from shutil import copyfile

VSC_DATA = os.getenv('VSC_DATA')
VSC_SCRATCH = os.getenv('VSC_SCRATCH')

def _create_model_training_folder(writer, files_to_same):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))

def eval_path(path):
    return path.replace("VSC_DATA", VSC_DATA).replace("VSC_SCRATCH", VSC_SCRATCH)