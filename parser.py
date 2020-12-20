import argparse

parser = argparse.ArgumentParser(description='Chest x-ray image model training')

# Training related hyper-parameters
parser.add_argument('--batch_size', default=16, type=int,
					help='Mini-batch size (default: 16)')
parser.add_argument('--init_lr', default=5e-3, type=float, 
        			help='Initial learning rate')
parser.add_argument('--num_train_epochs', default=300, type=int, 
        			help='Number of epochs to train for')
parser.add_argument('--scheduler', default='ReduceLROnPlateau', type=str, 
                    help='The scheduler for learning rate during training')
parser.add_argument('--model_architecture', default='resnet7_2_1', type=str,
                    help='Neural network architecture to be used')
parser.add_argument('--loss', default='CE', type=str,
                    help='Which loss function to use: CE, reweighted_CE')

parser.add_argument('--image_dir',
        			default='/data/vision/polina/scratch/ruizhi/chestxray/data/png_16bit',
        			help='Root directory of the CXR image data')
parser.add_argument('--output_dir',
        			default='/data/vision/polina/projects/chestxray/work_space_v2/'\
        			'joint_model_training/image_model/pretrain6/',
        			help='Directory for model training/evaluation output')
parser.add_argument('--data_split_path',
					default='/data/vision/polina/projects/chestxray/work_space_v2/report_processing/' \
	                'edema_labels-12-03-2019/mimic-cxr-sub-img-edema-split-manualtest.csv',
	                help='CSV path for data fold split')
parser.add_argument('--run_id',
					default=None, type=str,
                    help='An ID for this run')
parser.add_argument('--checkpoint_path',
                    default='/data/vision/polina/projects/chestxray/work_space_v2/'\
                            'joint_model_training/image_model/pretrain6/train1235_val4/'\
                            'checkpoints/checkpoints1/pytorch_model_epoch199.bin',
                    help='Path of the checkpoint to be evaluated ')
parser.add_argument('--eval_epoch', default=250,
                    help='The epoch that is used for evaluation')
parser.add_argument('--commit_sha', default=None,
                    type=str, help='Commit sha specified for evaluation')

parser.add_argument('--do_train', default=False, action='store_true', 
        			help='Whether to perform training')
parser.add_argument('--do_eval', default=False, action='store_true', 
        			help='Whether to perform evaluation')
parser.add_argument('--data_split_mode', default='cross_val', 
                    help='Whether to run in cross_val or testing mode')
parser.add_argument('--logging_steps', default=50, type=int, 
        			help='Number of steps for logging')
parser.add_argument('--training_folds', default=[1,2,3,4],
                    nargs='+', type=int, help="folds for training")
parser.add_argument('--evaluation_folds', default=[5],
                    nargs='+', type=int, help="folds for validation")
parser.add_argument('--image_format', default='png',
                    help='The format of the images that the model is reading, by default png')

def get_args(): return parser.parse_args()
