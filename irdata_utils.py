import numpy as np
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
import os

def dim2cuboid(dims, offset=np.zeros((3,1))):
        corners = np.array([[0, 0, 0],
                            [dims[0], 0, 0],
                            [dims[0], dims[1], 0],
                            [0, dims[1], 0],
                            [0, 0, dims[2]],
                            [dims[0], 0, dims[2]],
                            [dims[0], dims[1], dims[2]],
                            [0, dims[1], dims[2]]]) + np.tile(offset.T, (8,1))
        
        verts = [[corners[0], corners[1], corners[2], corners[3]],
                [corners[4], corners[5], corners[6], corners[7]],
                [corners[0], corners[1], corners[5], corners[4]],
                [corners[2], corners[3], corners[7], corners[6]],
                [corners[1], corners[2], corners[6], corners[5]],
                [corners[4], corners[7], corners[3], corners[0]]]
        
        return corners, verts

def plotatfmag(micpos, sig_gt, sig_pred, config, idx_plot_list=np.arange(5), figdir='', fname='', data_type='atf_mag'):
    '''
    sig: (M,L)
    srcpos: (M,3)
    '''
    if data_type == 'atf_mag':
        atf_mag_gt = sig_gt
        atf_mag_pred = sig_pred
    else:
        raise NotImplementedError
    #f_bin = np.linspace(0,config["max_frequency"],round(config["fft_length"]/2)+1)[1:]
    f_bin = np.arange(config["num_freq"]+1)[1:]/config["num_freq"]*config["fs"]/2
    micpos_np = micpos.to("cpu").detach().numpy().copy()

    plt.figure(figsize=(12,round(6*len(idx_plot_list))))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    for itr, idx_plot in enumerate(idx_plot_list):
        plt.subplot(len(idx_plot_list),1, itr+1)

        plt.plot(f_bin, atf_mag_gt[idx_plot,:].to('cpu').detach().numpy().copy(), label="Ground Truth", color='b',linestyle=':')
        plt.plot(f_bin, atf_mag_pred[idx_plot,:].to('cpu').detach().numpy().copy(), label="Predicted", color='b')
        plt.grid()
        plt.legend()
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        ylim = [-50,30]
        plt.ylim(ylim)
        plt.xlim([0,config["fs"]/2])
        plt.title(f"ATF ({micpos_np[idx_plot,0]:.2f} m, {micpos_np[idx_plot,1]:.2f} m, {micpos_np[idx_plot,2]:.2f} m)")

    if figdir == '':
        figdir = config["artifacts_dir"] + "/figure/atf/"
    if fname == '':
        raise NotImplementedError
    os.makedirs(figdir, exist_ok=True)
    plt.savefig(f"{fname}.pdf")
    plt.close()

def replace_activation_function(model, act_func_new, act_func_old=nn.ReLU):
    for child_name, child in model.named_children():
        if isinstance(child, act_func_old):
            setattr(model, child_name, act_func_new)
        else:
            replace_activation_function(child, act_func_new, act_func_old)

class MSE(th.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, data, target, dim=2, data_type='atf', mean=True):
        '''
        :param data:   (B,2,L,S) complex (or float) tensor
        :param target: (B,2,L,S) complex (or float) tensor
        :return: a scalar or (B,2,S) tensor
        '''
        #print(data.shape, target.shape)
        MSE = th.mean(th.abs(data - target).pow(2), dim=dim)
        if mean:
            MSE = th.mean(MSE)
        return MSE

class LSD(th.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, data, target, dim=2, data_type='atf_mag', mean=True):
        '''
        :param data:   (B,2,L,S) complex (or float) tensor
        :param target: (B,2,L,S) complex (or float) tensor
        :return: a scalar or (B,2,S) tensor
        '''
        #print(data.shape, target.shape)
        LSD = th.sqrt(th.mean((data - target).pow(2), dim=dim))
        if mean:
            LSD = th.mean(LSD)
        return LSD

class HelmholtzLossFreq(th.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, net, pos, k):
        return self._loss(net, pos, k)
    def _loss(self, net, pos, k):
        pos.requires_grad_(True)
        p = net.forward(pos=pos)['output_c']
        
        grads = th.autograd.grad(p, pos, grad_outputs=th.ones_like(p), create_graph=True)[0]
        lap = 0
        for i in range(pos.shape[1]):
            lap += th.autograd.grad(grads[:,i], pos, grad_outputs=th.ones_like(grads[:,i]), create_graph=True)[0][:,i]
        
        loss = th.mean(th.abs(lap + k**2 * p).pow(2))
        return loss  

class HelmholtzLoss(th.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, net, pos, k, freq):
        return self._loss(net, pos, k, freq)
    def _loss(self, net, pos, k, freq):
        pos.requires_grad_(True)
        p = net.forward(pos=pos, freq=freq)['output_c']
        
        loss = 0
        for ff in range(freq.shape[0]):
            grads = th.autograd.grad(p[:,ff], pos, grad_outputs=th.ones_like(p[:,ff]), create_graph=True)[0]
            lap = 0
            for i in range(pos.shape[1]):
                lap += th.autograd.grad(grads[:,i], pos, grad_outputs=th.ones_like(grads[:,i]), create_graph=True)[0][:,i]
        
            loss += th.mean(th.abs(lap + k[ff]**2 * p[:,ff]).pow(2))
        return loss  

class Net(nn.Module):
    def __init__(self, model_name="network", use_cuda=True):
        super().__init__()
        
        if th.cuda.is_available() and use_cuda:
                self.use_cuda = use_cuda
                self.device = th.device('cuda')
        else:
                self.use_cuda = False
                self.device = th.device('cpu')
        self.model_name = model_name

    def save(self, dir, suffix=""):
        '''
        save the network to model_dir/model_name.suffix.net
        :param model_dir: directory to save the model to
        :param suffix: suffix to append after model name
        '''
        if self.use_cuda:
                self.cpu()

        if suffix == "":
                fname = f"{dir}/{self.model_name}.net"
        else:
                fname = f"{dir}/{self.model_name}.{suffix}.net"

        th.save(self.state_dict(), fname)
        
        if self.use_cuda:
                self.cuda()
            
    def load_from_file(self, fname):
        '''
        load network parameters from fname
        :param fname: file containing the model parameters
        '''
        if self.use_cuda:
                self.cpu()

        states = th.load(fname)
        self.load_state_dict(states)

        if self.use_cuda:
                self.cuda()
        print(f"Loaded: {fname}")

    def load(self, dir, suffix=''):
        '''
        load network parameters from dir/model_name.suffix.net
        :param dir: directory to load the model from
        :param suffix: suffix to append after model name
        '''
        if suffix == "":
                fname = f"{dir}/{self.model_name}.net"
        else:
                fname = f"{dir}/{self.model_name}.{suffix}.net"
        self.load_from_file(fname)

    def num_trainable_parameters(self):
        '''
        :return: the number of trainable parameters in the model
        '''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class NewbobAdam(th.optim.Adam):
    def __init__(self,
                 weights,
                 net,
                 artifacts_dir,
                 initial_learning_rate=0.001,
                 decay=0.5,
                 max_decay=0.01,
                 timestamp = ""
                 ):
        '''
        Newbob learning rate scheduler
        :param weights: weights to optimize
        :param net: the network, must be an instance of type src.utils.Net
        :param artifacts_dir: (str) directory to save/restore models to/from
        :param initial_learning_rate: (float) initial learning rate
        :param decay: (float) value to decrease learning rate by when loss doesn't improve further
        :param max_decay: (float) maximum decay of learning rate
        '''
        super().__init__(weights, lr=initial_learning_rate, eps=1e-8)
        self.last_epoch_loss = np.inf
        self.second_last_epoch_loss = np.inf
        self.total_decay = 1
        self.net = net
        self.decay = decay
        self.max_decay = max_decay
        self.artifacts_dir = artifacts_dir
        self.timestamp = timestamp
        # store initial state as backup
        if decay < 1.0:
            net.save(artifacts_dir, suffix="newbob"+self.timestamp)
    
    def update_lr_two(self, loss):
        '''
        update the learning rate based on the current loss value and historic loss values
        :param loss: the loss after the current iteration
        '''
        if self.last_epoch_loss > self.second_last_epoch_loss and loss > self.last_epoch_loss and self.decay < 1.0 and self.total_decay > self.max_decay:
            self.total_decay = self.total_decay * self.decay
            print(f"NewbobAdam: Decay learning rate (loss degraded from { self.second_last_epoch_loss} to {loss})."
                  f"Total decay: {self.total_decay}")
            # restore previous network state
            self.net.load(self.artifacts_dir, suffix="newbob"+self.timestamp)
            # decrease learning rate
            for param_group in self.param_groups:
                param_group['lr'] = param_group['lr'] * self.decay
        else:
            self.second_last_epoch_loss = self.last_epoch_loss
            self.last_epoch_loss = loss
        # save last snapshot to restore it in case of lr decrease
        if self.decay < 1.0 and self.total_decay > self.max_decay:
            self.net.save(self.artifacts_dir, suffix="newbob"+self.timestamp)

    def update_lr_one(self, loss):
        '''
        update the learning rate based on the current loss value and historic loss values
        :param loss: the loss after the current iteration
        '''
        if loss > self.last_epoch_loss and self.decay < 1.0 and self.total_decay > self.max_decay:
            self.total_decay = self.total_decay * self.decay
            print(f"NewbobAdam: Decay learning rate (loss degraded from {self.last_epoch_loss} to {loss})."
                  f"Total decay: {self.total_decay}")
            # restore previous network state
            # self.net.load(self.artifacts_dir, suffix="newbob"+self.timestamp)
            # decrease learning rate
            for param_group in self.param_groups:
                param_group['lr'] = param_group['lr'] * self.decay
        self.last_epoch_loss = loss
        # else:
        #     self.last_epoch_loss = loss
        # save last snapshot to restore it in case of lr decrease
        # if self.decay < 1.0 and self.total_decay > self.max_decay:
        #     self.net.save(self.artifacts_dir, suffix="newbob"+self.timestamp)
    
    def update_lr_step(self, gamma=0.1):
        '''
        update the learning rate based on the current loss value and historic loss values
        :param loss: the loss after the current iteration
        '''
        for param_group in self.param_groups:
            param_group['lr'] = param_group['lr'] * gamma
        self.total_decay = self.total_decay * gamma
        print(f"NewbobAdam: Decay learning rate."
              f"Total decay: {self.total_decay}")