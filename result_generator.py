
import torch
from models import ResNet  # Assuming models.py is in the same directory
from metrics import *
from PIL import Image
import matplotlib.pyplot as plt
import time
import utils 
import os
from dense_to_sparse import *

def generate_results(model_name, current_timestamp):
    if model_name.strip() not in ["kitti-bs64", "nyuv2-bs8"]:
        return False
    
    # Set device to 'cuda' if you have a GPU, else 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Define the model parameters (the same as during training)
    # layers = 50  # Number of layers in ResNet (could be 18, 34, 50, 101, 152)
    # decoder_type = "deconv3"  # Decoder type (e.g., 'upconv', 'upproj', etc.)
    # output_size = (224, 224)  # Image size (height, width) of the model's output
    # in_channels = 4  # Number of input channels (3 for RGB images)

    # # Initialize the model
    # model = ResNet(layers=layers, decoder=decoder_type, output_size=output_size, in_channels=in_channels, pretrained=False)

    # # Load the checkpoint
    # checkpoint = torch.load(f'pretrained_models/{model_name.strip()}.pth.tar', map_location=device)
    # model.load_state_dict(checkpoint['model'].state_dict()) 
    # model.eval() 

    # preprocess the image and give it for evaluation
    # read the input images


    # hardcoded ahe.. ikde sagla ghe... jpeg n all
    rgb_image_path = f"static/uploads/rgb/rgb_{current_timestamp}.jpeg"
    sparse_depth_image_path = f"static/uploads/sparse-depth/sparse_depth_{current_timestamp}.jpeg"

    rgb_image = Image.open(rgb_image_path)
    sparse_image = Image.open(sparse_depth_image_path)

    # evaluation mode
    model_path = f'pretrained_models/{model_name.strip()}.pth.tar'
    checkpoint = torch.load(model_path, map_location=device)
    args = checkpoint['args']
    start_epoch = checkpoint['epoch'] + 1
    best_result = checkpoint['best_result']
    model = checkpoint['model']
    print("=> loaded best model (epoch {})".format(checkpoint['epoch']))

    args = {
        'arch' : 'resnet50',    # these are the choices : ['resnet18', 'resnet50]
        'data' : 'nyudepthv2',  # these are the choices : ['nyudepthv2', 'kitti']
        'modality' : 'rgbd',     # these are the choices : MyDataloader.modality_names -> ['rgb', 'rgbd', 'd']
        'num_samples' : 100,
        'max_depth' : 0.0,     # if the modality = rgb then the depth is 0 else it is -1.0
        'sparsifier' : UniformSampling.name, # these are the choices : [UniformSampling.name, SimulatedStereo.name] -> ['uar', 'sim_stereo']
        'decoder' : 'deconv3',   # these are the choices : Decoder.names -> ['deconv2', 'deconv3', 'upconv', 'upproj']
        'workers' : 10,
        'epochs' : 15,
        'criterion': 'l1', # these are the choices ['l1', 'l2']
        'batch_size' : 32,
        'lr' : 0.01,
        'momentum' : 0.9,
        'weight_decay' : 1e-4,
        'print_freq' : 10,
        'resume' : '',
        'evaluate' : '',
        'pretrained' : True,
    }

    # set to evaluation mode
    args['evaluate'] = True

    # set to kitti default is nyuv2
    if model_name.find("kitti") == 0:
        args['data'] = 'kitti'

    print(args)


    # _, val_loader = create_data_loaders(args)

    print("-------  Data loader created successfully  ")
    # print(val_loader)
    
    # validate(val_loader, model, checkpoint['epoch'], write_to_file=False)
    

    # return the generated result here
    
    return f"rgb_{current_timestamp}.{rgb_image_path.split('.')[-1]}"


def create_data_loaders(args):
    # Data loading code
    print("=> creating data loaders ...")
    traindir = os.path.join('datadir/', args['data'], 'train')
    valdir = os.path.join('datadir/', args['data'], 'val')
    train_loader = None
    val_loader = None

    # sparsifier is a class for generating random sparse depth input from the ground truth
    sparsifier = None
    max_depth = args['max_depth'] if args['max_depth'] >= 0.0 else np.inf
    if args['sparsifier'] == UniformSampling.name:
        sparsifier = UniformSampling(num_samples=args['num_samples'], max_depth=max_depth)
    elif args.sparsifier == SimulatedStereo.name:
        sparsifier = SimulatedStereo(num_samples=args['num_samples'], max_depth=max_depth)

    if args['data'] == 'nyudepthv2':
        from dataloaders.nyu_dataloader import NYUDataset
        if not args['evaluate']:
            train_dataset = NYUDataset(traindir, type='train',
                modality=args['modality'], sparsifier=sparsifier)
        val_dataset = NYUDataset(valdir, type='val',
            modality=args['modality'], sparsifier=sparsifier)

    elif args['data'] == 'kitti':
        from dataloaders.kitti_dataloader import KITTIDataset
        if not args['evaluate']:
            train_dataset = KITTIDataset(traindir, type='train',
                modality=args['modality'], sparsifier=sparsifier)
        val_dataset = KITTIDataset(valdir, type='val',
            modality=args['modality'], sparsifier=sparsifier)

    else:
        raise RuntimeError('Dataset not found.' +
                           'The dataset must be either of nyudepthv2 or kitti.')

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, shuffle=False, num_workers=args['workers'], pin_memory=True)

    # put construction of train loader here, for those who are interested in testing only
    if not args['evaluate']:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args['batch_size'], shuffle=True,
            num_workers=args['workers'], pin_memory=True, sampler=None,
            worker_init_fn=lambda work_id:np.random.seed(work_id))
            # worker_init_fn ensures different sampling patterns for each data loading thread

    print("=> data loaders created.")
    return train_loader, val_loader



def validate(val_loader, model, epoch, write_to_file=True):
    average_meter = AverageMeter()
    model.eval() # switch to evaluate mode
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(input)
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # save 8 images for visualization
        skip = 50
        if args.modality == 'd':
            img_merge = None
        else:
            if args.modality == 'rgb':
                rgb = input
            elif args.modality == 'rgbd':
                rgb = input[:,:3,:,:]
                depth = input[:,3:,:,:]

            if i == 0:
                if args.modality == 'rgbd':
                    img_merge = utils.merge_into_row_with_gt(rgb, depth, target, pred)
                else:
                    img_merge = utils.merge_into_row(rgb, target, pred)
            elif (i < 8*skip) and (i % skip == 0):
                if args.modality == 'rgbd':
                    row = utils.merge_into_row_with_gt(rgb, depth, target, pred)
                else:
                    row = utils.merge_into_row(rgb, target, pred)
                img_merge = utils.add_row(img_merge, row)
            elif i == 8*skip:
                filename = output_directory + '/comparison_' + str(epoch) + '.png'
                utils.save_image(img_merge, filename)

        
        print('Test: [{0}/{1}]\t'
                't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                'MAE={result.mae:.2f}({average.mae:.2f}) '
                'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                'REL={result.absrel:.3f}({average.absrel:.3f}) '
                'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))


    avg = average_meter.average()
    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    return avg, img_merge
