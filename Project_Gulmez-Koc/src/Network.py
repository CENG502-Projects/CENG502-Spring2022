import torch
import torch.nn as nn
from torchvision import models
from torchmetrics.functional import precision_recall

class SubNet(nn.Module):
    def __init__(self):    
        super(SubNet, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        layers = list(resnet50.children())
        self.lastStage = nn.Sequential(*layers[7:9])
        self.fullyConnected = nn.Linear(2048, 20)
        self.activation = torch.nn.Sigmoid()
        
    def forward(self,x):
        x = self.lastStage(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fullyConnected(x)
        x = self.activation(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Net(nn.Module):
    def __init__(self):    
        super(Net,self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        layers = list(resnet50.children())
        self.backbone = nn.Sequential(*layers[:7])
        self.subnetU = SubNet()
        self.subnetR = SubNet()

    def forward(self, x):
        x = self.backbone(x)
        u = self.subnetU(x)
        r = self.subnetR(x)
        return u,r    
            
    def num_flat_features(self, x):
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__=="__main__":

    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  
    device = torch.device(dev)  

    net = Net().to(device)
    print(net)
    input = torch.randn(1, 3, 224, 224).to(device)  # nSamples x nChannels x Height x Width
    print("input.shape",input.shape)
    out = net(input)
    print("out",out)

    ################# Test Loss ##########################
    Lcls = nn.BCEWithLogitsLoss()
    Lcon = nn.MSELoss()

    xU = torch.randn(1, 3, 224, 224).to(device)  # nSamples x nChannels x Height x Width
    xR = torch.randn(1, 3, 224, 224).to(device)  # nSamples x nChannels x Height x Width
    yU = torch.randn(1, 20).to(device)
    yR = torch.randn(1, 20).to(device)
    _lambda = 0.1

    u , uHat = net(xU)
    rHat , r = net(xR)
    loss = Lcls(u,yU) + Lcls(r,yR) + _lambda * ( Lcon(u,uHat) + Lcon(r,rHat))
    loss.backward()

    '''
    ################# Test Speed ##########################

    resnet50 = models.resnet50(pretrained=True).to(device)
    print(resnet50)
    out = resnet50(input)

    #GPU-WARM-UP
    for _ in range(10):
        _ = resnet50(input)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1
    timings=np.zeros((repetitions,1))
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = resnet50(input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)
    '''
