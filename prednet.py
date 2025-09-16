import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

class Prednet(nn.Module):
    def __init__(
        self,
        A_stack_sizes,
        R_stack_sizes,
        A_filter_sizes,
        Ahat_filter_sizes,
        R_filter_sizes,
        pixel_max=1,
        lstm_activation='relu',
        A_activation='relu',
        extrap_time=None,
        output_type='prediction',
        device='cpu'
    ):
        super(Prednet, self).__init__()
        self.A_stack_sizes = A_stack_sizes
        self.R_stack_sizes = R_stack_sizes
        self.A_filter_sizes = A_filter_sizes
        self.Ahat_filter_sizes = Ahat_filter_sizes
        self.R_filter_sizes = R_filter_sizes
        self.lstm_activation = lstm_activation
        self.A_activation = A_activation
        self.extrap_time = extrap_time
        self.output_type = output_type
        self.device = device
        self.pixel_max = pixel_max
        self.nb_layers = len(self.A_stack_sizes)
        
        # all conv layers for LSTM, A and Ahat predictions
        self.conv_layers = {c: [] for c in ['i', 'f', 'o', 'c', 'A', 'Ahat']}
        
        for c in self.conv_layers.keys():
            for l in range(self.nb_layers):
                if c == 'Ahat':
                    self.conv_layers[c].append(
                        nn.Conv2d(
                            in_channels=self.R_stack_sizes[l],
                            out_channels=self.A_stack_sizes[l],
                            kernel_size=self.Ahat_filter_sizes[l],
                            stride=(1,1),
                            padding = int((self.Ahat_filter_sizes[l] - 1) / 2) 
                        )
                    )

                    Ahat_act = 'relu' if l == 0 else self.A_activation
                    self.conv_layers[c].append(self.get_activation(Ahat_act))
                    
                elif c == 'A':
                    if self.is_not_top_layer(l):
                        self.conv_layers[c].append(
                            nn.Conv2d(
                                in_channels=self.A_stack_sizes[l]*2,
                                out_channels=self.A_stack_sizes[l+1],
                                kernel_size=self.A_filter_sizes[l],
                                stride=(1,1),
                                padding = int((self.A_filter_sizes[l] - 1) / 2) 
                            )
                        )
                        self.conv_layers[c].append(self.get_activation(self.A_activation))
                else:
                    # For LSTM layers ['i', 'f', 'o', 'c']
                    # The LSTM takes R_l^t-1, E_l^t-1, and R_l+1^t if not the top layer
                    in_channels = self.A_stack_sizes[l]*2 + self.R_stack_sizes[l]
                    if self.is_not_top_layer(l):
                        in_channels += self.R_stack_sizes[l+1]
                        
                    self.conv_layers[c].append(
                            nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=self.R_stack_sizes[l],
                                kernel_size=self.R_filter_sizes[l],
                                stride=(1,1),
                                padding = int((self.R_filter_sizes[l] - 1) / 2) 
                            )
                        )
                
        # Convert to Pytorch ModuleList and set as instance attribute
        for layer_name, conv_layer_list in self.conv_layers.items():
            self.conv_layers[layer_name] = nn.ModuleList(conv_layer_list)
            setattr(self, layer_name, self.conv_layers[layer_name])
        
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
    def get_activation(self, act):
        if act == 'relu':
            return nn.ReLU()
        if act == 'tanh':
            return nn.Tanh()

    def is_not_top_layer(self, l):
        return l != self.nb_layers - 1
    
    def hard_sigmoid(self, x):
        return torch.clamp(0.2 * x + 0.5, min=0.0, max=1.0)
    
    def batch_flatten(self, x):
        shape = list(x.size())
        dim = np.prod(shape[1:])
        dim = int(dim)
        return x.view(-1, dim)
    
    def get_initial_states(self, input_shape):
        batch, num_channels, input_height, input_width = input_shape
        
        R_list, c_list, E_list = [], [], []
        layer_names =  {s: self.nb_layers for s in ['R', 'c', 'E']}
        
        with torch.no_grad():
            for c in layer_names:
                for l in range(layer_names[c]):
                    downsample_fact = 2 ** l
                    h = input_height // downsample_fact
                    w = input_width // downsample_fact
                    
                    if c == 'R':
                        R_list.append(torch.zeros((batch, self.R_stack_sizes[l], h, w), device=self.device, dtype=torch.float32))
                    if c == 'c':
                        c_list.append(torch.zeros((batch, self.R_stack_sizes[l], h, w), device=self.device, dtype=torch.float32))
                    elif c == 'E':
                        E_list.append(torch.zeros((batch, self.A_stack_sizes[l]*2, h, w), device=self.device, dtype=torch.float32))
        
        states = {
          'R': R_list,
          'c': c_list, 
          'E': E_list,
        }    
    
        if self.extrap_time:
            with torch.no_grad():
                states['frame_prediction'] = torch.zeros(input_shape, device=self.device, dtype=torch.float32)
                states['timestep'] = 1
            
        return states
        
    def step(self, A, states):
        # states = {
        #   'R': R_list,
        #   'c': c_list, 
        #   'E': E_list,
        #   'frame_prediction': frame_prediction,
        #   'timestep': timestep
        # }
        R_current = states['R']
        E_current = states['E']
        c_current = states['c']
        
        if self.extrap_time:
            timestep = states['timestep']
            frame_prediction = states['frame_prediction']
            if timestep >= self.extrap_time:
                A = frame_prediction
                
        R_list, E_list, c_list = [], [], []
        R_upper = None
        
        # Top down pass
        for l in reversed(range(self.nb_layers)):
            # Inputs contain R_l^t-1, E_l^t-1, and R_l+1^t if not the top layer
            inputs = [R_current[l], E_current[l]]
            if self.is_not_top_layer(l):
                inputs.append(R_upper)
                
            inputs_torch = torch.cat(inputs, dim=-3)
            
            in_gate = self.hard_sigmoid(self.conv_layers['i'][l](inputs_torch))
            forget_gate = self.hard_sigmoid(self.conv_layers['f'][l](inputs_torch))
            out_gate = self.hard_sigmoid(self.conv_layers['o'][l](inputs_torch))
            cell_state = self.tanh(self.conv_layers['c'][l](inputs_torch))
            
            c_next = forget_gate * c_current[l] + in_gate * cell_state
            lstm_act = self.get_activation(self.lstm_activation)
            R_next = out_gate * lstm_act(c_next)
            
            c_list.insert(0, c_next)
            R_list.insert(0, R_next)
            
            if l > 0:
                R_upper = self.upsample(R_next)
        
        # Bottom up pass
        for l in range(self.nb_layers):
            # ReLU(Conv(R_l^t))
            Ahat = self.conv_layers['Ahat'][2*l](R_list[l])
            Ahat = self.conv_layers['Ahat'][2*l+1](Ahat)
            
            if l == 0:
                Ahat = torch.clamp(Ahat, max=self.pixel_max)
                frame_prediction = Ahat
            # print(Ahat.shape, A.shape)
            E_pos = self.relu(Ahat - A)
            E_neg = self.relu(A - Ahat)
            E_list.append(torch.cat([E_neg, E_pos], dim=-3))
            
            # TODO: Extract the outputs from certain module and layer
            
            if self.is_not_top_layer(l):
                # print(l, E_list[l].shape)
                A = self.conv_layers['A'][2*l](E_list[l])
                A = self.conv_layers['A'][2*l+1](A)
                A = self.pool(A)
        
        if self.output_type == 'prediction':
            output = frame_prediction
        else:
            for l in range(self.nb_layers):
                layer_error = torch.mean(self.batch_flatten(E_list[l]), dim=-1, keepdim=True)
                all_error = layer_error if l == 0 else torch.cat((all_error, layer_error), dim=-1)
            if self.output_type == 'error':
                output = all_error
            else:
                output = torch.cat((self.batch_flatten(frame_prediction), all_error), dim=-1)
                
        states = {
          'R': R_list,
          'c': c_list, 
          'E': E_list,
        }    
        
        if self.extrap_time:
            states['frame_prediction'] = frame_prediction
            states['timestep'] = timestep + 1
        
        return output, states
        
    def forward(self, A0_with_timesteps, initial_states):
        # change (batch, nt, c, h, w) to (nt, batch, c, h, w)
        A0_with_timesteps = A0_with_timesteps.transpose(0, 1)
        
        nt = A0_with_timesteps.size()[0]
        
        hidden_states = initial_states
        output_list = []
        
        for t in range(nt):
            A = A0_with_timesteps[t, ...]
            output, hidden_states = self.step(A, hidden_states)
            output_list.append(output)
        
        return output_list


if __name__ =='__main__':
    n_channels = 1
    img_height = 64
    img_width  = 64

    A_stack_sizes = (n_channels, 48, 96)
    R_stack_sizes = A_stack_sizes
    A_filter_sizes = (3, 3)
    Ahat_filter_sizes = (3, 3, 3)
    R_filter_sizes = (3, 3, 3)
    
    prednet = Prednet(
        A_stack_sizes=A_stack_sizes, 
        R_stack_sizes=R_stack_sizes, 
        A_filter_sizes=A_filter_sizes, 
        R_filter_sizes=R_filter_sizes, 
        Ahat_filter_sizes=Ahat_filter_sizes,
        pixel_max=1,
        lstm_activation='relu', 
        A_activation='relu', 
        extrap_time=None, 
        output_type='all'
    )
    print(prednet)
    
    
    