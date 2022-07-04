import torch
from torch import Tensor
from data import MINDatasetSampler
from fw_backbone import Conv4, Conv6, resnet10
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip
from model import RLAgent

def euclidean_distance(prototype_features: Tensor, query_features: Tensor) -> Tensor:
    nc, feature_dim = prototype_features.size()
    nq, feature_dim = query_features.size()
    prototype_features = prototype_features.unsqueeze(0).expand(nq, nc, feature_dim)
    query_features = query_features.unsqueeze(1).expand(nq, nc, feature_dim)
    return torch.sum(torch.square(prototype_features - query_features), dim=2)

def train_episode(backbone: torch.nn.Module,
                  data_sampler: MINDatasetSampler, 
                  num_class: int, 
                  num_support: int, 
                  num_query: int) -> Tensor:
    criterion = torch.nn.CrossEntropyLoss()
    support_set_list, query_set_list = data_sampler.random_sample_classes(num_class, num_support, num_query)
    prototype_list = []
    for support_set  in support_set_list:
        support_features = backbone(support_set.to("cuda"))
        prototype = torch.sum(support_features, dim=0) / num_support
        prototype_list.append(prototype)
    prototype_features = torch.stack(prototype_list, dim=0)

    loss = 0
    for index, query_set in enumerate(query_set_list):
        query_features = backbone(query_set.to("cuda"))
        distance = euclidean_distance(prototype_features,query_features)
        loss += criterion(-distance, torch.tensor(num_query*[index], device="cuda")) # scores = -distance
    return loss

def train_episode_attention(backbone: torch.nn.Module,
                            rl_agent: torch.nn.Module,
                            data_sampler: MINDatasetSampler, 
                            num_class: int, 
                            num_support: int, 
                            num_query: int,
                            agent_timesteps: int = 5,
                            test: bool = False) -> Tensor:
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    support_set_list, query_set_list = data_sampler.random_sample_classes(num_class, num_support, num_query)
    sf_list = []
    qf_list = []
    log_prob_list = []
    reward_list = []
    for t in range(agent_timesteps+1):
        if t == 0:
            for support_set  in support_set_list:
                support_features = backbone(support_set.to("cuda"))
                sf_list.append(support_features)
            
            for index, query_set in enumerate(query_set_list):
                query_features = backbone(query_set.to("cuda"))
                qf_list.append(query_features)
        else:
            prototype_list = []
            for i, support_set in enumerate(support_set_list):
                support_set = support_set.to("cuda")
                attention, _ = rl_agent(support_set, sf_list[0])
                support_features = backbone(support_set, attention)
                sf_list.append(support_features)
                sf_list.pop(0)
                prototype = torch.sum(support_features, dim=0) / num_support
                prototype_list.append(prototype)
            prototype_features = torch.stack(prototype_list, dim=0)

            for index, query_set in enumerate(query_set_list):
                query_set = query_set.to("cuda")
                attention, log_prob = rl_agent(query_set, qf_list[0])
                query_features = backbone(query_set, attention)
                qf_list.append(query_features)
                qf_list.pop(0)
                distance = euclidean_distance(prototype_features,query_features)
                reward = criterion(-distance, torch.tensor(num_query*[index], device="cuda")).unsqueeze(1) # scores = -distance
                reward_list.append(-reward)
                log_prob_list.append(log_prob)
    log_probs = torch.cat(log_prob_list, dim=0)
    returns = torch.cat(reward_list, dim=0)
    loss = (-log_probs * returns.detach()) -returns 
    return loss.mean()

def train(num_episodes : int,
          learning_rate : float,
          backbone: torch.nn.Module,
          rl_agent: torch.nn.Module,
          train_data_sampler: MINDatasetSampler,
          val_data_sampler: MINDatasetSampler,
          num_class: int, 
          num_support: int, 
          num_query: int,
          with_attention: bool = False) -> None: 
    if with_attention:
        optimizer = torch.optim.Adam(list(backbone.parameters()) + list(rl_agent.parameters()), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(backbone.parameters(), lr=learning_rate)

    best_val_loss = -float("inf")
    for i in range(num_episodes):
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        if with_attention:
            loss = train_episode(backbone, train_data_sampler, num_class, num_support, num_query)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss = train_episode_attention(backbone, rl_agent, val_data_sampler, num_class, num_support, num_query, agent_timesteps=5)
            loss.backward()
            optimizer.step()
        else:
            loss = train_episode(backbone, train_data_sampler, num_class, num_support, num_query)
            loss.backward()
            optimizer.step()

        if (i+1) % 10 == 0:
            with torch.no_grad():
                if with_attention:
                    val_loss = test_accuracy_attention(10, backbone, rl_agent, val_data_sampler, num_class, num_support, num_query, agent_timesteps=5).item()
                else:
                    val_loss = test_accuracy(10, backbone, val_data_sampler, num_class, num_support, num_query).item()
                if  best_val_loss < val_loss:
                    best_val_loss = val_loss
                    print("Current best vall loss:", best_val_loss, "at episode", i)
                    torch.save(backbone.state_dict(), "fe_best.pt".format(i))
                    if with_attention:
                        torch.save(rl_agent.state_dict(), "ra_best.pt".format(i))
    torch.save(backbone.state_dict(), "fe_last.pt")
    if with_attention:
        torch.save(rl_agent.state_dict(), "ra_last.pt")

        

def test_accuracy(num_episodes : int,
          backbone: torch.nn.Module,
          test_data_sampler: MINDatasetSampler,
          num_class: int, 
          num_support: int, 
          num_query: int) -> None:
    episode_accuracies = []
    with torch.no_grad():
        # backbone.eval()
        for _ in range(num_episodes):
            support_set_list, query_set_list = test_data_sampler.random_sample_classes(num_class, num_support, num_query)
            prototype_list = []
            for support_set  in support_set_list:
                support_features = backbone(support_set.to("cuda"))
                prototype = torch.sum(support_features, dim=0) / num_support
                prototype_list.append(prototype)
            prototype_features = torch.stack(prototype_list, dim=0)

            accuracy = 0
            for index, query_set in enumerate(query_set_list):
                query_features = backbone(query_set.to("cuda"))
                distance = euclidean_distance(prototype_features,query_features)
                predicted_class_index = torch.argmax(-distance, dim=1) # scores = -dist
                accuracy += (predicted_class_index == index).sum()
            episode_accuracies.append(accuracy*100/(num_class*num_query))
    return torch.mean(torch.tensor(episode_accuracies))

def test_accuracy_attention(num_episodes : int,
                            backbone: torch.nn.Module,
                            rl_agent: torch.nn.Module,
                            test_data_sampler: MINDatasetSampler,
                            num_class: int, 
                            num_support: int, 
                            num_query: int,
                            agent_timesteps: int = 5) -> None:

    episode_accuracies = []
    with torch.no_grad():
        for _ in range(num_episodes):
            support_set_list, query_set_list = test_data_sampler.random_sample_classes(num_class, num_support, num_query)
            sf_list = []
            qf_list = []
            pc_list = []
            for t in range(agent_timesteps+1):
                if t == 0:
                    for support_set  in support_set_list:
                        support_features = backbone(support_set.to("cuda"))
                        sf_list.append(support_features)
                    
                    for index, query_set in enumerate(query_set_list):
                        query_features = backbone(query_set.to("cuda"))
                        qf_list.append(query_features)
                else:
                    prototype_list = []
                    for i, support_set in enumerate(support_set_list):
                        support_set = support_set.to("cuda")
                        attention, _ = rl_agent(support_set, sf_list[0], True)
                        support_features = backbone(support_set, attention)
                        sf_list.append(support_features)
                        sf_list.pop(0)
                        prototype = torch.sum(support_features, dim=0) / num_support
                        prototype_list.append(prototype)
                    prototype_features = torch.stack(prototype_list, dim=0)

                    for index, query_set in enumerate(query_set_list):
                        query_set = query_set.to("cuda")
                        attention, _ = rl_agent(query_set, qf_list[0], True)
                        query_features = backbone(query_set, attention)
                        qf_list.append(query_features)
                        qf_list.pop(0)
                        if t == agent_timesteps:
                            distance = euclidean_distance(prototype_features,query_features)
                            predicted_class_index = torch.argmax(-distance, dim=1) # scores = -dist
                            pc_list.append(predicted_class_index)
            accuracy = 0
            for true_class, predicted_class in enumerate(pc_list):
                accuracy += (predicted_class == true_class).sum()
            episode_accuracies.append(accuracy*100/(num_class*num_query))
    return torch.mean(torch.tensor(episode_accuracies))

def main(with_attention, num_class, num_support, num_query, num_episodes, learning_rate, backbone):
    transforms = Compose([
        RandomResizedCrop(size=(84,84)),
        RandomHorizontalFlip()
    ])
    print("With attention:", with_attention, ", Backbone:", type(backbone).__name__, 
          ", num_class:", num_class, ", num_support:", num_support, ", num_query:", num_query, ", num_episodes:", num_episodes)
    backbone = backbone.to("cuda")
    if not with_attention:
        train_data_sampler = MINDatasetSampler(["images/train", "images/val"], transform=transforms, read_all=True, device="cpu")
        val_data_sampler = MINDatasetSampler(["images/val"], transform=transforms, read_all=True, device="cpu")
        train(num_episodes, learning_rate, backbone, None, train_data_sampler, val_data_sampler, num_class, num_support, num_query, False)
        test_data_sampler = MINDatasetSampler(["images/test"], transform=transforms, read_all=True, device="cuda")
        backbone.load_state_dict(torch.load("fe_best.pt"))
        print(test_accuracy(num_episodes//2, backbone, test_data_sampler, num_class, num_support, num_query))
    else:
        train_data_sampler = MINDatasetSampler(["images/train"], transform=transforms, read_all=True, device="cpu")
        val_data_sampler = MINDatasetSampler(["images/val"], transform=transforms, read_all=True, device="cpu")
        with torch.no_grad():
            test = backbone(torch.randn((1,3,84,84), requires_grad=False, device="cuda"), 1)
            rl_agent = RLAgent(84, [64, 64, 64], 2, test.size(1), backbone.size, backbone.channel).to("cuda")
        train(num_episodes, learning_rate, backbone, rl_agent, train_data_sampler, val_data_sampler, num_class, num_support, num_query, True)
        test_data_sampler = MINDatasetSampler(["images/test"], transform=transforms, read_all=True, device="cuda")
    
        backbone.load_state_dict(torch.load("fe_best.pt"))
        rl_agent.load_state_dict(torch.load("ra_best.pt"))
        print(test_accuracy_attention(num_episodes//2, backbone, rl_agent, test_data_sampler, num_class, num_support, num_query, 5))

if __name__ == "__main__":
    main(with_attention=True, 
         num_class=5, 
         num_support=1, 
         num_query=16, 
         num_episodes=1200, 
         learning_rate=1e-3, 
         backbone=resnet10(attention_layer=2))