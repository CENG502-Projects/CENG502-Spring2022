from turtle import back
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import model

class Trainer:
    def __init__(self, config):
        self.num_classes = config["num_classes"]  # Number of classificated classes. Set according to dataset
        self.transform = config["transform"]
        # self.train_set = config["train_set_dir"]
        # self.test_set = config["test_set_dir"]
        self.batch_size = config["batch_size"]
        self.finetune_epoch = config["finetune_epoch"]  # Set in case of finetuning for backbone

        use_pretrained = config["use_pretrained"]  # Decide to use pretrained, if so, set backbone_dir
        backbone_dir = config["backbone_dir"]  
        self.device = config["device"]
        self.lr = config["learning_rate"]
        self.T = config["T"]  # Number of reinforce steps for one pass of batch
        self.epoch = config["epoch"]  # Training epoch for RAP
        self.image_res = config["image_res"]  # Input image resolution
        self.attention_layer = config["attention_layer"]  # Decide after which layer to apply attention
        self.alpha = config["alpha"]  # Adjust the weight of validation set loss for reward
        self.model_save = config["model_save"]  # Model saving frequency
        agent_dir = config["agent_dir"]  # Directory of trained RL agent

        # Create backbone
        if use_pretrained: self.backbone = model.resnet18(True, load_path=backbone_dir, num_classes = self.num_classes, attention_layer = self.attention_layer)
        else: self.backbone = model.resnet18(False, num_classes = self.num_classes, attention_layer = self.attention_layer)
        self.backbone = self.backbone.to(self.device)

        # Get attention sizes to create RL agent
        test_img = torch.randn((self.batch_size, 3, self.image_res, self.image_res)).to(self.device)
        self.embed_feature_res = self.backbone.fc.in_features
        self.attention_map_size = self.backbone(test_img, out_layer=self.attention_layer).size()
        
        # Create RL agent
        self.agent = model.RLAgent(self.image_res, 
                                embed_feature_res = self.embed_feature_res,
                                attention_res = self.attention_map_size[2], 
                                attention_channels = self.attention_map_size[1]).to(self.device)

        # Load if pretrained RL agent exists
        if agent_dir != "": self.agent.load_state_dict(torch.load(agent_dir))
            
        # Load sets and create dataloader
        self.train_set = torchvision.datasets.CIFAR10("CIFAR10", True, transform=transform, download=True)
        # Method requires validation set. Split train by 40k to 10k
        self.train_set, self.val_set = random_split(self.train_set, [40000, 10000])
        self.test_set = torchvision.datasets.CIFAR10("CIFAR10", False, transform=transform, download=True) 
        
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size)
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size)
    
    def set_parameter_requires_grad(self, model, feature_extracting): 
        """This function freezes the model if mode is feature_extracting. Taken from PyTorch"""
        req_grad = True
        if feature_extracting: req_grad = False
        for param in model.parameters():
            param.requires_grad = req_grad

    def train(self):
        """Train the RL and Backbone for supervised classification"""
        optim_agent = torch.optim.Adam(self.agent.parameters(), self.lr, betas=[0.9, 0.99])
        freeze_backbone = False

        # No need since optim_backbone is not created if backbone is freezed
        # However, updating last fc layer is an ablation option.
        # In case just last fc is updated, update set_parameter_requires_grad according to it.
        # Follow torch official tutorial.
        params_to_update = self.backbone.parameters()
        if freeze_backbone: 
            self.set_parameter_requires_grad(self.backbone, True)  # Freeze backbone
            for _,param in self.backbone.named_parameters():
                params_to_update = []
                if param.requires_grad == True:
                    params_to_update.append(param)
        if not freeze_backbone: optim_bakcbone = torch.optim.Adam(params_to_update, self.lr, betas=[0.9, 0.99])

        criterion = nn.CrossEntropyLoss()
        curr_eval = self.eval(self.val_loader, self.backbone, self.agent)
        print(curr_eval)
        self.backbone.train()
        self.agent.train()
        prev_eval = 0.0
        for epoch in range(self.epoch):
            running_loss_train = 0.0
            running_loss_val = 0.0
            for inputs, labels in self.train_loader:
                optim_agent.zero_grad()
                if not freeze_backbone: optim_bakcbone.zero_grad()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.backbone(inputs)
                if not freeze_backbone:
                    train_loss = criterion(outputs, labels)
                    running_loss_train += train_loss.item() * inputs.size(0)
                    train_loss.backward()
                # Train rl agent on validation set
                val_inps, val_labels = next(iter(self.train_loader))
                val_inps = val_inps.to(self.device)
                val_labels = val_labels.to(self.device)
                self.backbone(val_inps)  # Updates resnet18.embedded_feature
                rein_loss = 0.0
                for _ in range(self.T):
                    attention_map, log_prob = self.agent(val_inps, self.backbone.embedded_feature)
                    val_outputs = self.backbone(val_inps, attention_map)  # backbone embed features are updated
                    reward = -self.alpha * criterion(val_outputs, val_labels)
                    rein_loss += -(log_prob.mean() * reward.detach()) - reward/self.T
                running_loss_val += rein_loss.item() * val_inps.size(0)
                rein_loss.backward()

                if not freeze_backbone: optim_bakcbone.step()
                optim_agent.step()
                # Change according to freeze_backbone # print('{} Loss at {}: Train Loss = {:.4f}, Rein Loss = {:.4f}'.format("Train", epoch, train_loss.item(), running_loss_val))
            running_loss_train = running_loss_train / len(self.train_loader.dataset)
            running_loss_val = running_loss_val / len(self.train_loader.dataset)
            epoch_loss = running_loss_val+running_loss_train
            print('{} Loss at {}: {:.6f}, Train: {:.4f}, Rein: {:.6f}'.format("Train", epoch, epoch_loss, running_loss_train, running_loss_val))
            curr_eval = self.eval(self.val_loader, self.backbone, self.agent)
            print(curr_eval)
            if curr_eval > prev_eval:
                torch.save(self.backbone.state_dict(), "models/loss_rap_resnet18_rein.pt".format(epoch))            
                torch.save(self.agent.state_dict(), "models/loss_rap_agent_rein.pt".format(epoch)) 
                prev_eval = curr_eval

                
    def finetune_backbone(self, backbone, feature_extracting=True, first_loop=False):
        # Followed https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
        
        if first_loop:  # Prevent adding new layer at each time.
            self.set_parameter_requires_grad(backbone, feature_extracting)  # Decide which is going to be trained
                                                                            # just last channel or whole backbone
            num_ftrs = backbone.fc.in_features  # Get previous feature layer size
            backbone.fc = nn.Linear(num_ftrs, 10)
            backbone.fc = backbone.fc.to(self.device)

        # optim = torch.optim.SGD(backbone.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.01, nesterov=True)
        optim = torch.optim.Adam(backbone.parameters(), lr=self.lr, betas=[0.5, 0.99])
        criterion = nn.CrossEntropyLoss()
        print("Starting training")
        backbone.train()
        prev_eval = 0.0
        for epoch in range(self.finetune_epoch):
            running_loss = 0.0

            for inputs, labels in self.train_loader:
                optim.zero_grad()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = backbone(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optim.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(self.train_loader.dataset)
            
            print('{} Loss at {}: {:.4f}'.format("Train", epoch, epoch_loss))
            curr_eval = self.eval(self.val_loader, backbone, None)
            train_eval = self.eval(self.train_loader, backbone, None)
            print(curr_eval, train_eval)
            if curr_eval > prev_eval:
                torch.save(backbone.state_dict(), "models/finetuned_resnet18.pt".format(epoch))
                prev_eval = curr_eval
            # if epoch % self.model_save == 0: torch.save(backbone.state_dict(), "models/finetuned_resnet18_{}.pt".format(epoch))            

    def eval(self, test_loader, backbone, agent=None):
        correct = 0
        # backbone.eval()
        # if not agent is None: agent.eval()

        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for inputs, labels in test_loader:
                labels = labels.to(self.device)
                inputs = inputs.float().to(self.device)
                outputs = backbone(inputs)
                if agent:
                    for _ in range(self.T):
                        attention_map, _ = agent(inputs, backbone.embedded_feature)
                        outputs = backbone(inputs, attention_map)
                # print(criterion(outputs, labels))
                predicted_label = torch.argmax(outputs, dim=1)
                correct += torch.sum(predicted_label == labels)
        return correct/len(test_loader.dataset)
        
if __name__=="__main__":
    # For torch pretrained models mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    image_res = 32
    transform = transforms.Compose([transforms.PILToTensor(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(image_res),
                                    transforms.ConvertImageDtype(torch.float),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

    config = {
        "transform" : transform,
        "num_classes" : 10,
        "batch_size" : 128,
        "finetune_epoch" : 100,
        "use_pretrained" : True,
        "agent_dir": "models/loss_rap_agent_rein.pt",
        "backbone_dir" : "models/loss_rap_resnet18_rein.pt",
        "device" : "cuda",
        "learning_rate": 1e-6,  # 1e-6 in the paper
        "epoch": 4000,
        "T": 5,
        "image_res": image_res,
        "attention_layer": 3,
        "alpha": 1e-1,  # 1e-4 paper value
        "model_save": 10
    }

    trainer = Trainer(config=config)
    trainer.train()
    
    # trainer.finetune_backbone(trainer.backbone, False, True)

    # performance = trainer.eval(trainer.test_loader, trainer.backbone, trainer.agent)
    # print(performance)
    
    # first_loop = True
    # while performance < 92.6:
    #     print(performance)
    #     trainer.finetune_backbone(trainer.backbone, False, first_loop=first_loop)
    #     performance = trainer.eval(trainer.test_loader, trainer.backbone)
    #     first_loop = False