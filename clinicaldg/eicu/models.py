import torch
from torch import nn
import clinicaldg.eicu.Constants as Constants

class FlattenedDense(nn.Module):
    def __init__(self, ts_cat_levels, static_cat_levels, emb_dim, num_layers, num_hidden_units,
                            t_max = 48, dropout_p = 0.2):   
        super().__init__()      
        self.ts_cat_levels = ts_cat_levels
        self.static_cat_levels = static_cat_levels
        self.emb_dim = emb_dim
        
        self.ts_embedders = nn.ModuleList([nn.Embedding(num_embeddings = ts_cat_levels[i], embedding_dim = emb_dim) for i in ts_cat_levels])
        self.static_embedders = nn.ModuleList([nn.Embedding(num_embeddings = static_cat_levels[i], embedding_dim = emb_dim) for i in static_cat_levels])
        
        input_size = (len(Constants.ts_cont_features) * t_max + len(Constants.ts_cat_features) * emb_dim * t_max 
                         +  len(Constants.static_cont_features) + len(Constants.static_cat_features) * emb_dim             
                     )        
        
        layers = [nn.Linear(input_size, num_hidden_units)]
        for i in range(1, num_layers):
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p = dropout_p))
            layers.append(nn.BatchNorm1d(num_hidden_units))
            layers.append(nn.Linear(num_hidden_units, num_hidden_units))
        
        
        self.clf = nn.Sequential(*layers)
        self.n_outputs = num_hidden_units
        
    
    def forward(self, x):
        ts_cont_feats, ts_cat_feats, static_cont_feats, static_cat_feats = (x['ts_cont_feats'].float(),
                x['ts_cat_feats'], x['static_cont_feats'].float(), x['static_cat_feats'])
        # shape of ts inputs: (batch_size, 48, n_features)
        # shape of static inputs: (batch_size, n_features)        
        ts_cont_feats = ts_cont_feats.flatten(start_dim = 1) # now (batch_size, n_features*48)
        
        cat_embs = []
        for i in range(len(self.ts_embedders)):
            cat_embs.append(self.ts_embedders[i](ts_cat_feats[:, :, i]).flatten(start_dim = 1))
        
        for i in range(len(self.static_embedders)):
            cat_embs.append(self.static_embedders[i](static_cat_feats[:, i]))
        
        x_in = torch.cat(cat_embs, dim = 1)
        x_in = torch.cat([x_in, ts_cont_feats, static_cont_feats], dim = 1)
        
        return self.clf(x_in)
    
    
class GRUNet(nn.Module):
    def __init__(self, ts_cat_levels, static_cat_levels, emb_dim, num_layers, num_hidden_units,
                            t_max = 48, dropout_p = 0.2):   
        super().__init__()      
        self.ts_cat_levels = ts_cat_levels
        self.static_cat_levels = static_cat_levels
        self.emb_dim = emb_dim
        self.t_max = t_max
        
        self.ts_embedders = nn.ModuleList([nn.Embedding(num_embeddings = ts_cat_levels[i], embedding_dim = emb_dim) for i in ts_cat_levels])
        self.static_embedders = nn.ModuleList([nn.Embedding(num_embeddings = static_cat_levels[i], embedding_dim = emb_dim) for i in static_cat_levels])
                
        input_size = (len(Constants.ts_cont_features)  + len(Constants.ts_cat_features) * emb_dim  
                         +  len(Constants.static_cont_features) + len(Constants.static_cat_features) * emb_dim             
                     )        
        
        self.gru = nn.GRU(input_size = input_size, hidden_size = num_hidden_units, num_layers = num_layers, 
                          batch_first = True, dropout = dropout_p, bidirectional = True)
        
        self.n_outputs = num_hidden_units * 2 # bidirectional
        
    
    def forward(self, x):
        ts_cont_feats, ts_cat_feats, static_cont_feats, static_cat_feats = (x['ts_cont_feats'].float(),
                x['ts_cat_feats'], x['static_cont_feats'].float(), x['static_cat_feats'])
        # shape of ts inputs: (batch_size, 48, n_features)
        # shape of static inputs: (batch_size, n_features)     
        x_in = torch.cat([ts_cont_feats] + [embedder(ts_cat_feats[:, :, c]) for c, embedder in enumerate(self.ts_embedders)], dim = -1)    
            
        cat_embs = []     
        for i in range(len(self.static_embedders)):
            cat_embs.append(self.static_embedders[i](static_cat_feats[:, i]))
                
        statics = torch.cat([static_cont_feats] + cat_embs, dim = -1)
        statics = statics.unsqueeze(1).expand(statics.shape[0], self.t_max, statics.shape[-1])
        
        x_in = torch.cat([x_in, statics], dim = -1)
        
        return self.gru(x_in)[0][:, -1, :]