import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, LayerNorm, Linear
from torch_geometric.nn import global_mean_pool, global_max_pool
from src.models.fsgn_model import get_mlp_layers



def get_gnn_layers(num_conv_layers: int, hidden_channels, num_inp_features:int, 
                 gnn_layer, activation=nn.ReLU, normalization=None, dropout = None):
    """Creates GNN layers"""
    layers = nn.ModuleList()

    for i in range(num_conv_layers):
        if i == 0:
            layers.append(gnn_layer(num_inp_features, hidden_channels[i]))
            layers.append(activation())
            if normalization is not None:
                layers.append(normalization(hidden_channels[i]))
        else:
            layers.append(gnn_layer(hidden_channels[i-1], hidden_channels[i]))
            layers.append(activation())
            if normalization is not None:
                layers.append(normalization(hidden_channels[i]))

    return nn.ModuleList(layers)



class GNN(torch.nn.Module):
    def __init__(self, in_features, num_classes, hidden_channels, activation, normalization, num_conv_layers=3, layer='gcn',
                 use_input_encoder=True, encoder_features=128, apply_batch_norm=True,
                 apply_dropout_every=True, task='sex_prediction', use_scaled_data=False, dropout = 0):
        super(GNN, self).__init__()

        assert task in ['age_prediction', 'sex_prediction',  'weight_prediction', 'height_prediction', 'bmi_prediction']
        torch.manual_seed(12345)
        
        self.fc = torch.nn.ModuleList()
        self.task = task
        self.layer_type = layer
        self.use_input_encoder = use_input_encoder
        self.apply_batch_norm = apply_batch_norm
        self.dropout = dropout
        self.normalization_bool = normalization
        self.activation = activation
        self.apply_dropout_every = apply_dropout_every
        self.use_scaled_data = use_scaled_data

        if self.normalization_bool:
            self.normalization = LayerNorm
        else:
            self.normalization = None

        if self.use_input_encoder :
            self.input_encoder = get_mlp_layers(
                channels=[in_features, encoder_features],
                activation=nn.ELU,
            )
            in_features = encoder_features

        if layer == 'gcn':
            self.layers = get_gnn_layers(num_conv_layers, hidden_channels, num_inp_features=in_features,
                                        gnn_layer=GCNConv,activation=activation,normalization=self.normalization )
        elif layer == 'sageconv':
            self.layers = get_gnn_layers(num_conv_layers, hidden_channels,in_features,
                                        gnn_layer=SAGEConv,activation=activation,normalization=self.normalization )
        elif layer == 'gat':
            self.layers = get_gnn_layers(num_conv_layers, hidden_channels,in_features,
                                        gnn_layer=GATConv,activation=activation,normalization=self.normalization )
        
        for i in range((len(hidden_channels)-num_conv_layers)):
            self.fc.append(Linear(hidden_channels[i+num_conv_layers-1], hidden_channels[i+num_conv_layers]))

            

        #if apply_batch_norm:
        #    self.batch_layers = nn.ModuleList(
        #        [nn.BatchNorm1d(hidden_channels) for i in range(num_conv_layers)]
        #    )


        if task == 'sex_prediction':
            self.pred_layer = Linear(hidden_channels[len(hidden_channels)-1], num_classes)
        else:
            self.pred_layer = Linear(hidden_channels[len(hidden_channels)-1], 1)

        # print(self.layers)
        # print(self.fc)
        # print(self.pred_layer)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.use_input_encoder:
            x = self.input_encoder(x)

        if self.normalization is None:
            for i, layer in enumerate(self.layers):
                # Each GCN consists 2 modules GCN -> Activation 
                # GCN send edge index
                if i% 2 == 0:
                    x = layer(x, edge_index)
                else:
                    x = layer(x)

                if self.apply_dropout_every:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            for i, layer in enumerate(self.layers):
                # Each GCN consists 3 modules GCN -> Activation ->  Normalization 
                # GCN send edge index
                if i% 3 == 0:
                    x = layer(x, edge_index)
                else:
                    x = layer(x)

                if self.apply_dropout_every:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                

        # 2. Readout layer
        x = global_max_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)

       
        for i in range(len(self.fc)):
           x = self.fc[i](x)
           x = torch.tanh(x)
           x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.pred_layer(x)

        if self.use_scaled_data or self.task =='sex_prediction':
            x = torch.nn.Sigmoid()(x)
        
        return x