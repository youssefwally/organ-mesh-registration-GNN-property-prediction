import sys
import os
import ast
import wandb


CUR_USER = os.getlogin()
if CUR_USER == 'koksal':
    sys.path.append('/u/home/koksal/organ-mesh-registration-and-property-prediction/')
elif CUR_USER == 'wyo':
    sys.path.append('/u/home/wyo/final_integration/organ-mesh-registration-and-property-prediction/')
elif CUR_USER == 'manu':
    sys.path.append('FILL THIS WITH YOUR USERDIRECTORY NAME')

import torch
from torch import nn
from src.models.fsgn_model import MeshSeg
from src.data.organs_dataset import OrganMeshDataset
from torch_geometric.data import DataLoader
from tqdm import tqdm
from time import sleep
import mlflow
from src.models.baseline_model import GNN
import argparse
from torchmetrics import R2Score
from copy import deepcopy
def train(net, train_data, optimizer, loss_fn, device):
    """Train network on training dataset."""
    net.train()
    cumulative_loss = 0.0
    for data in train_data:
        data = data.to(device)
        optimizer.zero_grad()
        out = net(data)
        loss = loss_fn(out.squeeze(1), data.y.float())
        loss.backward()
        cumulative_loss += loss.item()
        optimizer.step()
    return cumulative_loss / len(train_data)

def calculate_val_loss(net, val_data, loss_fn, device):
    net.eval()
    cumulative_loss = 0.0
    for data in val_data:
        data = data.to(device)
        out = net(data)
        loss = loss_fn(out.squeeze(1), data.y.float())
        cumulative_loss += loss.item()
    return cumulative_loss / len(val_data)

def accuracy(predictions, gt_class_labels):
    """Compute accuracy of predicted segmentation labels.

    Parameters
    ----------
    predictions: [|V|, num_classes]
        Soft predictions of sex prediction 
    gt_class_labels: [|V|]
        Ground truth sex labels.
    Returns
    -------
    float
        Accuracy of predicted segmentation labels.    
    """
    #
    predicted_class_labels = torch.nn.Sigmoid()(predictions)
    predicted_class_labels = torch.round(predicted_class_labels)
    
    if predicted_class_labels.shape != gt_class_labels.shape:
        raise ValueError("Expected Shapes to be equivalent")
    correct_assignments = (predicted_class_labels == gt_class_labels).sum()
    num_assignemnts = predicted_class_labels.shape[0]
    return float(correct_assignments / num_assignemnts)


def evaluate_performance(dataset, net, configs, task='classification'):
    """Evaluate network performance on given dataset.

    Parameters
    ----------
    dataset: DataLoader
        Dataset on which the network is evaluated on.
    net: torch.nn.Module
        Trained network.
    device: str
        Device on which the network is located.

    Returns
    -------
    float:x
        Mean accuracy of the network's prediction on
        the provided dataset.
    """
    device = configs.device
    prediction_accuracies = []
    for data in dataset:
        data = data.to(device)
        predictions = net(data)
        if task == 'classification':
            prediction_accuracies.append(accuracy(predictions.squeeze(1), data.y))
        elif task == 'regression':
            if configs.eval_method == 'r2':
                measure_score = R2Score().to(device)
            elif configs.eval_method == 'mse':
                measure_score = torch.nn.MSELoss().to(device)
            elif configs.eval_method == 'mae':
                measure_score = torch.nn.L1Loss().to(device)
            prediction_accuracies.append(measure_score(predictions.squeeze(1), data.y))
        
    return sum(prediction_accuracies) / len(prediction_accuracies)

@torch.no_grad()
def test_classification(net, train_data, test_data, configs):
    net.eval()
    train_acc = evaluate_performance(train_data, net, configs)
    test_acc = evaluate_performance(test_data, net, configs)
    return train_acc, test_acc


@torch.no_grad()
def test_regression(net, train_data, test_data, configs):
    net.eval()
    train_score = evaluate_performance(train_data, net, configs, task='regression')
    test_score = evaluate_performance(test_data, net, configs, task='regression')
    return train_score, test_score

def build_optimizer(network, optimizer, learning_rate, weight_decay):
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(network.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def build_network(configs):
    """Builds network that is specified in the configs file.

    Args:
        configs (ArgParse Config file): Config file

    Returns:
        net: Pytorch Module
    """
    activation = getattr(nn, configs.activation)

    # FeastNet Model
    if configs.model=='fsgnet':
        model_params = dict(
        use_input_encoder = configs.use_input_encoder,
        num_classes=configs.num_classes,
        in_features=configs.in_features, 
        encoder_features=configs.enc_feats,
        conv_channels=configs.hidden_channels,
        activationf=activation,
        encoder_channels=[configs.enc_feats],
        decoder_channels=[256],
        num_heads=configs.num_heads,
        apply_batch_norm=configs.norm,
        use_scaled_data = configs.use_scaled_data,
        task = configs.task,  
        dropout = configs.dropout
    ) 

        net = MeshSeg(**model_params)

    # Baseline models that use GCN, GraphSAGE, GAT layers
    elif configs.model=='baseline':
        
        print('Baseline Model is initialized')
        model_params = dict(
        use_input_encoder = configs.use_input_encoder,
        num_classes=configs.num_classes,
        in_features=configs.in_features, 
        encoder_features = configs.enc_feats,
        hidden_channels= configs.hidden_channels,
        activation=activation,
        normalization = configs.norm,
        layer = configs.layer,
        num_conv_layers = configs.num_conv_layers,
        use_scaled_data = configs.use_scaled_data,
        task = configs.task,
        dropout = configs.dropout)
        net = GNN(**model_params)

    return net


def build_dataset(config, return_dataset=False):
    # Build Dataset

    train_dataset = OrganMeshDataset(config, mode='train')
    val_dataset = OrganMeshDataset(config, mode='val',)
    test_dataset = OrganMeshDataset(config, mode='test')
                                    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,  shuffle=False)

    if return_dataset:
        return train_dataset, val_dataset
    return train_loader, test_loader

def training_function(config=None):
    
    # note that we define values from `wandb.config` instead of 
    # defining hard values
    print('Training function config ',config)
    device = config.device
    print('Current Device',device)

    #Data Loader
    train_loader, test_loader = build_dataset(config, False)

    #Network
    print(device)
    net = build_network(config).to(device)
    
    #Optimizer
    optimizer = build_optimizer(net, config.optimizer, config.lr, config.weight_decay)

    #Loss Function
    if config.task == 'sex_prediction':
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        if config.loss == 'mse':
            loss_fn = torch.nn.MSELoss()
        elif config.loss == 'mae':
            loss_fn = torch.nn.L1Loss()

    best_test_acc = 0
    best_test_score = 0

    with tqdm(range(config.max_epoch), unit="Epoch") as tepochs:
        for epoch in tepochs:
            train_loss = train(net, train_loader, optimizer, loss_fn, device)
            val_loss = calculate_val_loss(net, test_loader, loss_fn, device)
            wandb.log({'train_loss': train_loss, 'val_loss':val_loss, 'epoch': epoch})
            
            if config.task == 'sex_prediction':
                train_acc, test_acc = test_classification(net, train_loader, test_loader, config)
                wandb.log({'train_acc': train_acc, 'test_acc':test_acc, 'epoch': epoch})

                tepochs.set_postfix(
                train_loss=train_loss,
                val_loss = val_loss,
                train_accuracy=100 * train_acc,
                test_accuracy=100 * test_acc,
                )
                sleep(0.1)

            # Regression Task: BMI prediction, Age prediction, Height prediction, Weight prediction
            else:
                train_score, test_score = test_regression(net, train_loader, test_loader, config)
                wandb.log({'train_score': train_score, 'test_score':test_score, 'epoch': epoch})

                tepochs.set_postfix(
                train_loss=train_loss,
                val_loss = val_loss,
                train_score = train_score,
                test_score = test_score)
                sleep(0.1)
            
            wandb.watch(net)

            # Logs the best evaluation scores and save the models
            if config.task == 'sex_prediction':
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    wandb.run.summary["best_test_acc"] = 100 *best_test_acc
                    wandb.run.summary["best_train_acc"] = 100 * train_acc
                    savedir = f'/u/home/{CUR_USER}/organ-mesh-registration-and-property-prediction/models/'
                    savedir = os.path.join(savedir, str(wandb.run.name))
                    if  not os.path.exists(savedir):
                        os.makedirs(savedir)
                    torch.save({'model':  deepcopy(net.state_dict()), 
                                'config': {k:v
                                for k,v in config.items()} }, f"{savedir}/classification_organ_{config.organ}_enc_channels_{config.hidden_channels}_best_testacc_{test_acc:.2f}.pth")

            # Regression tasks
            else:
                if config.eval_method == 'r2':
                    if test_score > best_test_score:
                        best_test_score = test_score
                        wandb.run.summary["best_test_score"] = test_score
                        wandb.run.summary["best_train_score"] = train_score
                        savedir = f'/u/home/{CUR_USER}/organ-mesh-registration-and-property-prediction/models/'
                        savedir = os.path.join(savedir, str(wandb.run.name))
                        if  not os.path.exists(savedir):
                            os.makedirs(savedir)
                        torch.save({'model': deepcopy(net.state_dict()),  
                                    'config': {k:v
                                    for k,v in config.items()} }, f"{savedir}/regression_organ_{config.organ}_enc_channels_{config.hidden_channels}_best_testr2_{round(best_test_score,2)}.pth")
                
                else:
                    #Lower is better in regression
                    if test_score < best_test_score:
                        best_test_score = test_score
                        wandb.run.summary["best_test_score"] = test_score
                        wandb.run.summary["best_train_score"] = train_score
                        savedir = f'/u/home/{CUR_USER}/organ-mesh-registration-and-property-prediction/models/'
                        savedir = os.path.join(savedir, str(wandb.run.name))
                        if  not os.path.exists(savedir):
                            os.makedirs(savedir)
                        torch.save({'model': deepcopy(net.state_dict()),  
                                    'config': {k:v
                                    for k,v in config.items()} }, f"{savedir}/regression_organ_{config.organ}_enc_channels_{config.hidden_channels}_best_testr2_{round(best_test_score,2)}.pth")

    if config.task == 'sex_prediction':
        print('Best Test Accuracy is ',best_test_acc)
    elif config.task == 'age_prediction':
        print('Best Test R2 score is ',best_test_score)
    


def build_args():
    parser = argparse.ArgumentParser(description='GNN for Organ Meshes')
    parser.add_argument("--model", type=str, default="baseline")
    parser.add_argument("--device", type=int, default=6)
    parser.add_argument("--max_epoch", type=int, default=50,
                        help="number of training epochs")
    parser.add_argument("--enc_feats", type=int, default=64,
                        help="Encoder features")        
    parser.add_argument("--num_heads", type=int, default=12,
                        help="number of hidden attention heads")

    parser.add_argument("--hidden_channels", nargs='+', type = int, default=[512, 512, 256, 256, 128, 64, 1],
                        help="Hidden dim of baseline")

    parser.add_argument("--num_train_samples", type=int, default=3000,
                        help="Number of training samples")  
    parser.add_argument("--num_test_samples", type=int, default=300,
                            help="Number of training samples")                        

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--in_features", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--num_conv_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.002)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--activation", type=str, default="ReLU")
    parser.add_argument("--norm", type=ast.literal_eval, default=True)

    parser.add_argument("--layer", type=str, default="gcn")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--use_input_encoder", type=ast.literal_eval, default=True)
    #parser.add_argument("--hparam_search", type=bool, default=False)
    parser.add_argument("--organ", type=str, default="liver")
    parser.add_argument("--task", type=str, default="sex_prediction")
    parser.add_argument("--use_registered_data", type=ast.literal_eval, default=True)
    parser.add_argument("--decimation_path", type=str, default="/data0/practical-wise2223/organ_mesh/organ_decimations_ply/")
    parser.add_argument("--registeration_path", type=str, default="/data0/practical-wise2223/organ_mesh/gendered_organ_registrations_ply/")
    parser.add_argument("--split_path", type=str, default='/data0/practical-wise2223/organ_mesh/data/')
    parser.add_argument("--root", type=str, default='/data0/practical-wise2223/organ_mesh/organ_meshes')
    parser.add_argument("--basic_feat_path", type=str, default='/data0/practical-wise2223/organ_mesh/basic_features.csv')
    parser.add_argument("--use_scaled_data", type=ast.literal_eval, default=False)
    parser.add_argument("--bridge_path", type=str, default='/data0/practical-wise2223/organ_mesh/Bridge_eids_60520_87802.csv')
    parser.add_argument("--return_dataset", type=ast.literal_eval, default=False)
    parser.add_argument("--loss", type=str, default='mae')
    parser.add_argument("--eval_method", type=str, default='mae')
    
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = build_args()
    #print('Args : ',args)

    device = args.device #if args.device >= 0 else "cpu"

    if args.device != 'cuda' and args.device != 'cpu':
        device = int(args.device)
        
    print('Usual training starts')
    run = wandb.init(
    project="mesh_gnn_organ_presentation",
    notes="baseline",
    tags=[args.model, args.organ, args.task, args.layer, f'enc_feats_{args.enc_feats}', f'heads_{args.num_heads}', f'hidden_channels_{args.hidden_channels}', f'num_layers_{args.num_conv_layers}'],
    config=args,
    )

    wandb.config.update( {'device':device }, allow_val_change=True)
    #wandb.config.device = device

    #wdb_config = wandb.config
    print('WDB CONFIG ',wandb.config)
    training_function(wandb.config)
