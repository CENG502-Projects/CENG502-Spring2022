import torchvision
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from tqdm import tqdm
import cv2
import torch
import json
from models.hrnet import HRNet
from torch.nn import Embedding
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn.models import GAT
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero
import pickle
from numba import njit
from sklearn.metrics import average_precision_score as aps
from poseval import evaluateAP
import argparse

MLP_hidden_dimension = 100

use_attention = True

device = torch.device(0)
pose_model = HRNet(32, 17, 0.1).to(device)      
pose_model.load_state_dict(torch.load("pose_hrnet_w32_384x288.pth"))

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, MLP_hidden_dimension)
        self.hidden_fc = nn.Linear(MLP_hidden_dimension, 50)
        self.output_fc = nn.Linear(50, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        
        x = x.view(batch_size, -1)

        h_1 = F.relu(self.input_fc(x))

        h_2 = F.relu(self.hidden_fc(h_1))

        y_pred = self.output_fc(h_2)

        return y_pred, h_2

device = torch.device("cuda:0")
batch_size = 1

model_human_detection = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

width = 384
height = 288
heatmap_width = 96
heatmap_height = 72

def crop_and_rescale_boxes_with_annots(img, box, joints_list):
    cropped_image = None
    resized_and_scaled_joints = None
    defined_joints = np.array(joints_list)                   

    box[0][0] = max(0, box[0][0])
    box[0][1] = max(0, box[0][1])

    tmp_image = img[box[0][1]:box[1][1],box[0][0]:box[1][0]]

    if tmp_image.shape[0]==0 or tmp_image.shape[1]==0:
        return [], []

    width_scale_ratio = height/tmp_image.shape[1] 
    height_scale_ratio = width/tmp_image.shape[0] 

    tmp_image = cv2.resize(tmp_image, (height,width))
    cropped_image = tmp_image

    if len(defined_joints)<1:
        return cropped_image, []

    defined_joints[:, 1] = defined_joints[:, 1] - box[0][0] 
    defined_joints[:, 2] = defined_joints[:, 2] - box[0][1] 


    defined_joints[:, 1] = defined_joints[:, 1]*width_scale_ratio  
    defined_joints[:, 2] = defined_joints[:, 2]*height_scale_ratio  

    resized_and_scaled_joints = defined_joints
            
    return cropped_image, resized_and_scaled_joints

@njit   
def generate_heatmaps(ground_truth, bbox):    
    width_scale_ratio = heatmap_height/bbox[2]
    height_scale_ratio = heatmap_width/bbox[3]
    heat_map = np.zeros((heatmap_width, heatmap_height))
    for i in range(heatmap_height):
        for j in range(heatmap_width):
            distance_x = (i - (ground_truth[0]-bbox[0])*width_scale_ratio)
            distance_y = (j - (ground_truth[1]-bbox[1])*height_scale_ratio)
            norm = distance_x * distance_x + distance_y * distance_y
            
            val = np.exp(-((norm)/9))
            heat_map[j, i] = val
    return heat_map

def find_in_json(json_obj, attribute ,key):
    for i in range(len(json_obj)):
        if json_obj[i][attribute] == key:
            return json_obj[i]

def find_in_all_jsons(json_obj, attribute ,key):
    all_jsons = []
    for i in range(len(json_obj)):
        if json_obj[i][attribute] == key:
            all_jsons.append(json_obj[i])
    return all_jsons

class HeatmapLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask=None):
        assert pred.size() == gt.size()
        loss = ((pred - gt)**2)
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
        return loss

def human_pose_loss(output, target):
    loss = 0
    subtraction = torch.subtract(output, target)
    subtraction = torch.square(subtraction)
    subtraction = torch.sum(subtraction)
    loss = subtraction/(output.shape[0]*output.shape[1]*output.shape[2]*output.shape[3])
    return loss

def generate_global_edges(layer_num):
    edge_list = []
    prev_start_index = None
    curr_start_index = 17
    if layer_num <= 0:
        return None
    else:
        curr_start_index *= layer_num
        prev_start_index = curr_start_index-17

        for prev_ind in range(prev_start_index, prev_start_index+17):
            for curr_ind in range(curr_start_index, curr_start_index+17):
                edge_list.append((prev_ind, curr_ind))
    
    return edge_list

def create_edge_representations(layer_num):
    edge_list = []

    nose_adjaceny = np.array([[0,1],[1,0],[0,2],[2,0],[0,3],[3,0],[0,4],[4,0]])
    head_bottom_adjaceny = np.array([[1,2],[2,1],[1,5],[5,1]])
    head_top_adjaceny = np.array([[2,3],[3,2],[2,4],[4,2]])
    left_shoulder_adjaceny = np.array([[5,7],[7,5]])
    right_shoulder_adjaceny = np.array([[6,8],[8,6]])

    left_elbow_adjaceny = np.array([[7,9],[9,7]])
    right_elbow_adjaceny = np.array([[8,10],[10,8]])
    left_hip_adjaceny = np.array([[11,13],[13,11]])
    right_hip_adjaceny = np.array([[12,14],[14,12]])

    left_knee_adjaceny = np.array([[13,15],[15,13]])
    right_knee_adjaceny = np.array([[14,16],[16,14]])
    edge_list.append((
        nose_adjaceny,
        head_bottom_adjaceny,
        head_top_adjaceny,
        left_shoulder_adjaceny,
        right_shoulder_adjaceny,
        left_elbow_adjaceny,
        right_elbow_adjaceny,
        left_hip_adjaceny,
        right_hip_adjaceny,
        left_knee_adjaceny,
        right_knee_adjaceny
    ))

    new_edge_list = []
    for i in range(len(edge_list)):
        for j in range(len(edge_list[i])):
            new_edge_list.extend(edge_list[i][j]+17*layer_num)

    return new_edge_list

def create_adjaceny_matrix(matrix):
    adjaceny_matrix = np.zeros((45,45))
    for m in matrix:
        adjaceny_matrix[m[0],m[1]]=1
    return adjaceny_matrix

class GCNLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out)

    def forward(self, node_feats, adj_matrix):
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
        node_feats = self.projection(node_feats)
        node_feats = torch.bmm(adj_matrix, node_feats)
        node_feats = node_feats / num_neighbours
        return node_feats

feature_dimension = 2000
gnn_feature_dimension = feature_dimension
num_hidden_channels = 2000
gnn_out_channel_num = 2000

class GCNNet(torch.nn.Module):
    def __init__(self):
        super().__init__()        
        self.gnn_nodes = GAT(gnn_feature_dimension, num_hidden_channels, 2, out_channels=gnn_out_channel_num)

    def forward(self, x, edge_index):
        x = self.gnn_nodes(x, edge_index)
        return x

position_MLP = MLP(3, feature_dimension)
visual_feature_MLP = MLP(heatmap_width*heatmap_height,feature_dimension)
type_embed = Embedding(17, feature_dimension)

def process_frame_joints(joints, joint1_heatmap):
    joint1_heatmap = joint1_heatmap.view(17, -1)
    joints1 = torch.tensor(joints)#[0]
    joint1_xcoords = joints1[0::3]
    joint1_ycoords = joints1[1::3]
    joint1_is_visible = joints1[2::3]
    joint1_coord_pairs = torch.reshape(torch.stack((joint1_xcoords, joint1_ycoords, joint1_is_visible), axis=1),(17, 3))

    pos = position_MLP(joint1_coord_pairs.float())[0]
    vis = visual_feature_MLP(joint1_heatmap.float())[0]
    embed = type_embed(torch.from_numpy(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])))
    concatenated = torch.cat((pos,embed,vis), axis=1)
    Jk = (pos+vis+embed)/3
    return Jk

def create_dictionary_data(Jk, global_adjacency, local_adjacency):
    data = HeteroData()
    data["frame1"].x = Jk[0]
    data["frame2"].x = Jk[1]
    data["frame3"].x = Jk[2]

    data['frame1', 'f12', 'frame2'].edge_index = torch.from_numpy(np.array(global_adjacency[0]).T)%17
    data['frame2', 'f23', 'frame3'].edge_index = torch.from_numpy(np.array(global_adjacency[1]).T)%17
    data['frame1', 'self_f', 'frame1'].edge_index = torch.from_numpy(np.array(local_adjacency[0]).T)%17
    data['frame2', 'self_f', 'frame2'].edge_index = torch.from_numpy(np.array(local_adjacency[1]).T)%17
    data['frame3', 'self_f', 'frame3'].edge_index = torch.from_numpy(np.array(local_adjacency[2]).T)%17

    return data

def generate_prediction_dictionary(keys, bbox_head, track_id):
    point_list = []
    for i in range(len(keys)):
        if keys[i][2]==0:
            keys[i][0] = 0
            keys[i][1]
            pass
        point_list.append({
            "id":[i],
            "x":[keys[i][0]],
            "y":[keys[i][1]],
            "score":[1.0]
            })
    point_dict = {"point": point_list}
    annotation_dict = {
        "annopoints": [point_dict],        
        "x1":[bbox_head[0]],
        "x2":[bbox_head[0]+bbox_head[2]],
        "y1":[bbox_head[1]],
        "y2":[bbox_head[1]+bbox_head[3]],
        "track_id":[track_id],
        "score":[1.0]
        }
    
    correct_dict = {
        "annorect": [annotation_dict], 
        }
    
    annotations_list = [{"annolist": [correct_dict], "seq_id": 0, "annorect":[annotation_dict]}]
    return annotations_list

def generate_predictions(outputs, keypoints, bbox_head, track_id):
    gtFramesAll = generate_prediction_dictionary(keypoints, bbox_head, track_id)
    prFramesAll = generate_prediction_dictionary(outputs, bbox_head, track_id)
    return gtFramesAll, prFramesAll

def rescale_bbox(bbox_head, bbox):
    if bbox[2]==0 or bbox[3]==0:
        return np.array([0,0,0,0])
    bbox_head[:2] = bbox_head[:2] - bbox[:2]

    width_scale_ratio = heatmap_height/bbox[3]
    height_scale_ratio = heatmap_width/bbox[2]

    bbox_head[0] = bbox_head[0]*width_scale_ratio
    bbox_head[1] = bbox_head[1]*height_scale_ratio
    bbox_head[2] = bbox_head[2]*heatmap_width/bbox[2]
    bbox_head[3] = bbox_head[3]*heatmap_height/bbox[3]

    return bbox_head

def visual_feature_maps(backbone_model, image_name, joint_bbox):
    read_image1 = cv2.imread(image_name)

    read_image1 = read_image1[
        int(joint_bbox[0]):int(joint_bbox[0]+joint_bbox[2]),
        int(joint_bbox[1]):int(joint_bbox[1]+joint_bbox[3])
    ]
    if read_image1.shape[0]==0 or read_image1.shape[1]==0:
        return []

    read_image1 = torch.from_numpy(cv2.resize(read_image1, (width, height))).unsqueeze(0)
    read_image1 = read_image1.permute(0, 3, 1, 2).float()

    x, x1, x2, x3 = backbone_model(read_image1)
    concatenated_stages = torch.cat((x1[0], x2[0], x3[0]), 1)
    return concatenated_stages

def bounding_box_shape(image_shape):
    return [0, 0, image_shape[0], image_shape[1]]

def test_loop(gnn_model_name, trainloader):
    model = torch.load(gnn_model_name)

def train_gnn(dataset):
    pred_MLP = MLP(gnn_out_channel_num,3).to(device)
    backbone_model = HRNet()

    train_loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)

    gnn_model = GCNNet()
    if use_attention:
        node_types = ['frame1', 'frame2', 'frame3']
        edge_types = [
            ('frame1', 'f12', 'frame2'),
            ('frame2', 'f23', 'frame3'),
            ('frame1', 'self_f', 'frame1'),
            ('frame2', 'self_f', 'frame2'),
            ('frame3', 'self_f', 'frame3'),
        ]
        metadata = (node_types, edge_types)

        gnn_model = to_hetero(gnn_model, metadata)

    error = torch.nn.MSELoss()
    
    global_adjaceny1 = generate_global_edges(1)
    global_adjaceny2 = generate_global_edges(2)

    self_adjaceny1 = create_edge_representations(0)
    self_adjaceny2 = create_edge_representations(1)
    self_adjaceny3 = create_edge_representations(2)
    avg_pool = torch.nn.AvgPool1d(3)

    lr = 0.0001

    optimizer = torch.optim.Adam(
        list(gnn_model.parameters())+
        list(position_MLP.parameters())+
        list(visual_feature_MLP.parameters())+
        list(type_embed.parameters())+
        list(pred_MLP.parameters())
        , lr=lr)

    reduce_learning_rate = False
    is_validation = False
    for epoch in range(2000):
        total_score = 0
        with tqdm(train_loader) as tbar:
            if reduce_learning_rate:
                if epoch==5 or epoch==8:
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr']/10

            min_counter = 10000
            min_index = 0
            min_outputs = None
            min_keypoints = None
            total_score = 0
            moving_loss = 0
            for i, (images_1, images_2, images_3,images_4, joints) in enumerate(tbar):
                vis_list = []
                pos_list = []
                embed_list = []
                for k in range(len(joints)):
                    joint1_bbox = joints[k]["bbox1"]
                    joint2_bbox = joints[k]["bbox2"]
                    joint3_bbox = joints[k]["bbox3"]
                    joint4_bbox = joints[k]["bbox4"]

                    use_144_channels=False
                    if use_144_channels:
                        Jk1 = joints[k]["Jk1"][0]
                        Jk2 = joints[k]["Jk2"][0]
                        Jk3 = joints[k]["Jk3"][0]   

                        joint1_heatmap = visual_feature_maps(backbone_model, images_1[0], joint1_bbox)
                        joint2_heatmap = visual_feature_maps(backbone_model, images_2[0], joint2_bbox)
                        joint3_heatmap = visual_feature_maps(backbone_model, images_3[0], joint3_bbox)
                        joint4_heatmap = visual_feature_maps(backbone_model, images_4[0], joint4_bbox)
                    else:
                        joint1_heatmap = joints[k]["heatmaps1"]*255
                        joint2_heatmap = joints[k]["heatmaps2"]*255
                        joint3_heatmap = joints[k]["heatmaps3"]*255
                        joint4_heatmap = joints[k]["heatmaps4"]*255

                        if len(joint1_heatmap)==0 or len(joint2_heatmap)==0 or len(joint3_heatmap)==0 or len(joint4_heatmap)==0:
                            continue

                        Jk1 = process_frame_joints(joints[k]["keypoints1"][0], joint1_heatmap)
                        Jk2 = process_frame_joints(joints[k]["keypoints2"][0], joint2_heatmap)
                        Jk3 = process_frame_joints(joints[k]["keypoints3"][0], joint3_heatmap)

                    bbox_head = rescale_bbox(joints[k]["bbox_head3"][0].numpy(), joints[k]["bbox3"])
                    if bbox_head[2] == 0 or bbox_head[3]==0:
                        continue
                    
                    optimizer.zero_grad()
                    # Forward propagation
                    if use_attention:
                        data = create_dictionary_data([Jk1,Jk2,Jk3], [global_adjaceny1, global_adjaceny2], [self_adjaceny1, self_adjaceny2, self_adjaceny3])
                        outputs = gnn_model(data.x_dict, data.edge_index_dict)[('frame3', 'self_f', 'frame3')].to(device)
                    else:
                        Jk = torch.concat((Jk1, Jk2, Jk3))
                        adjacency = global_adjaceny1+self_adjaceny1+self_adjaceny2+self_adjaceny3
                        adjacency = np.array(adjacency).flatten().reshape((2, -1))
                        adjacency = torch.from_numpy(adjacency).long()

                        outputs = gnn_model(Jk, adjacency)[-17:].to(device)

                    outputs = pred_MLP(outputs)[0]
    
                    keypoints = np.zeros((17, 3)).astype(np.float32)    

                    for j in range(joints[k]["heatmaps3"].shape[1]):
                        key_point_outputs = np.unravel_index(joints[k]["heatmaps3"][0][j].argmax(), joints[k]["heatmaps3"][0][j].shape)
                        keypoints[j]=[key_point_outputs[1], key_point_outputs[0], joints[k]["keypoints3"][0][2+3*j]]
                    loss = error(outputs, torch.from_numpy(keypoints).to(device))
                    generate_train_images = True
                    if generate_train_images:
                        heat_out = np.zeros((17, heatmap_width, heatmap_height))
                        for z in range(17):
                            heat_out[z] = generate_heatmaps((int(outputs[z][0]), int(outputs[z][1])), [0,0,heatmap_height, heatmap_width])

                    loss.backward()
                    # Update parameters
                    optimizer.step()

                    outputs[-1] = torch.nn.functional.sigmoid(outputs[-1])
                    
                    gtFramesAll, prFramesAll = generate_predictions(np.round(outputs.cpu().detach().numpy()), keypoints, bbox_head, joints[k]["track_id"])
                    
                    score = evaluateAP.evaluateAP(gtFramesAll, prFramesAll, None, False, False)  

                    total_score=(total_score*i+score[0][-1])/(i+1)

                    print_best = False
                    if print_best:
                        if i%5==0:
                            print(total_score)

                    moving_loss = (moving_loss+loss.item())/(i+1)
                    tbar.set_postfix(loss=moving_loss, training_accuracy = total_score)
            save_model = True
            if save_model:
                torch.save({
                    "gnn_model":gnn_model,
                    "position_MLP":position_MLP,
                    "visual_feature_MLP":visual_feature_MLP,
                    "type_embed":type_embed,
                    "pred_MLP":pred_MLP
                    },
                    "gnn_models/"+str(epoch)+".pt"
                    
                    )
            if 0:
                print("For epoch ", str(epoch) , ": ",np.round(np.array(min_outputs.cpu().detach().numpy())).astype(np.int32), min_keypoints)

def train(model, train_loader):
    model = HRNet()
    model = model.to(device)
    
    loss_list = []
    iteration_list = []
    accuracy_list = []
    total_loss_list = []
    validation_total_loss_list = []

    total_training_accuracy = []
    error = human_pose_loss
    
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(200):
        try:
            os.mkdir("box_outputs/"+str(epoch))
        except:
            pass

        with tqdm(train_loader) as tbar:
            for i, (images, tmp_heat_maps) in enumerate(tbar):

                train_image = torch.from_numpy(np.moveaxis(np.array(images), -1, 1))

                train_image = train_image.to(device)

                tmp_heat_maps = tmp_heat_maps.to(device)
                optimizer.zero_grad()
                outputs = model(train_image).to(device)

                tmp_heat_maps = tmp_heat_maps.to(device)

                loss = error(outputs, tmp_heat_maps)
                loss.backward()
                optimizer.step()

                if (i+1)%batch_size == 0:
                    loss_list.append(loss.data.cpu())
                box_outputs = outputs.cpu().detach().numpy()
                tmp_heat_maps = tmp_heat_maps.cpu().detach().numpy()

                keypoints = np.zeros(box_outputs[0].shape[1:])

                for j in range(box_outputs[0].shape[0]):
                    key_point_outputs = np.unravel_index(box_outputs[0][j].argmax(), box_outputs[0][j].shape)
                    keypoints[key_point_outputs]=255

                cv2.imwrite("box_outputs/"+str(epoch)+"/pred_"+"_"+str(i)+".png", np.sum(box_outputs[0], axis=0)*255)

                cv2.imwrite("box_outputs/"+str(epoch)+"/thresholded_pred_"+"_"+str(i)+".png", keypoints)

                cv2.imwrite("box_outputs/"+str(epoch)+"/label_"+"_"+str(i)+".png", np.sum(tmp_heat_maps[0], axis=0)*255)

                train_image = np.moveaxis(np.array(train_image[0].cpu().detach().numpy()), 0, -1)

                cv2.imwrite("box_outputs/"+str(epoch)+"/img_"+str(epoch)+"_"+str(i)+".png", train_image*255)

                tbar.set_postfix(loss=loss.item())

class GNNDataset(Dataset):
    def __init__(self, image_npy, joint_root, annotation_npy, mode):
        self.names_npy = os.listdir(os.path.join(image_npy, mode))
        self.image_root = image_npy
        self.joint_root = joint_root
        self.annotation_root = annotation_npy
        self.mode = mode
    def __getitem__(self, idx):
        image = np.load(os.path.join(self.image_root, self.mode, self.names_npy[idx]), allow_pickle=True)
        joint = np.load(os.path.join(self.joint_root, self.mode, self.names_npy[idx]), allow_pickle=True)    
        
        human_list = []
        for k in range(len(joint)):
            human_list.append({
                "track_id":joint[k]["track_id"], "bbox1":joint[k]["bbox1"], "bbox2":joint[k]["bbox2"], "bbox3":joint[k]["bbox3"], "bbox4":joint[k]["bbox4"], \
                "keypoints1":joint[k]["keypoints1"], "keypoints2":joint[k]["keypoints2"], "keypoints3":joint[k]["keypoints3"], "keypoints4":joint[k]["keypoints4"], \
                "heatmaps1":joint[k]["heatmaps1"], "heatmaps2":joint[k]["heatmaps2"], "heatmaps3":joint[k]["heatmaps3"], "heatmaps4":joint[k]["heatmaps4"], \
                "bbox_head1":joint[k]["bbox_head1"],"bbox_head2":joint[k]["bbox_head2"], "bbox_head3":joint[k]["bbox_head3"], "bbox_head4":joint[k]["bbox_head4"], \
            })    
        return image[0], image[1], image[2], image[3], human_list

    def __len__(self):
        return (len(self.names_npy))

class PoseTrackDataset(Dataset):
    def read_and_process_init(self, image_npy, annotation_npy):
        self.names_npy = os.listdir(image_npy)
        self.image_root = image_npy
        self.annotation_root = annotation_npy

    def __init__(self, image_folder_path, all_annotation_paths, mode, transform=None):

        self.image_folder_path = image_folder_path

        self.file_names = []
        self.history_files = []
        self.history_file_joints = []

        self.annotations = []
        self.annotation_names = []
        self.all_heat_maps = []

        self.images = []
        self.heatmaps = []

        annotation_file_paths = os.listdir(os.path.join(all_annotation_paths, mode))
        for path in tqdm(annotation_file_paths):
            path_to_file = os.path.join(all_annotation_paths, mode, path)
            file_obj = open(path_to_file)
            annotation_file = json.load(file_obj)

            for annotation in annotation_file["annotations"]:
                
                image_path = find_in_json(annotation_file["images"], "id", annotation["image_id"])["file_name"]
                image_path = os.path.join(image_folder_path, image_path)

                if os.path.isfile(image_path) and "bbox" in annotation.keys():
                    if(annotation["bbox"][2] and annotation["bbox"][3])!=0:
                        self.file_names.append(image_path)
                        self.annotations.append(annotation)
                else:
                    continue
            
            counter=1
            tmp_history = []
            tmp_history_joints = []
            for images in annotation_file["images"]:
                image_id = images['id']
                image_path = os.path.join("dataset/posetrack_data", find_in_json(annotation_file["images"], "id", image_id)["file_name"])
                found_anno = find_in_all_jsons(annotation_file["annotations"], "image_id", image_id)

                if found_anno is None or not os.path.isfile(image_path):
                    continue
                keys = found_anno
                tmp_history.append(images["file_name"])
                if keys==[]:
                    continue
                else:
                    tmp_history_joints.append(keys)
                    counter+=1
                self.history_files.append(images["file_name"])
                self.history_file_joints.append(keys)

    def get_hrnet_joint_embeddings(self, image_name, bbox):
        global pose_model
        image = cv2.imread(os.path.join(self.image_folder_path, image_name))
        image = image[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
        if image.shape[0]==0 or image.shape[1]==0:
            return None
        image = cv2.resize(image, (height, width))
        image = torch.from_numpy(image)
        image = torch.permute(image, (2,0,1)).unsqueeze(0).float().to(device)
        predicted_heatmap = pose_model(image)[0].cpu().detach().numpy()
        return predicted_heatmap

    def get_joint_embeddings(self, joints, bbox):
        joint1_heatmap = np.zeros((17, heatmap_width, heatmap_height))

        if bbox is None or bbox[3]==0 or bbox[2]==0:
            return joint1_heatmap

        width_scale_ratio = heatmap_height/bbox[2]
        height_scale_ratio = heatmap_width/bbox[3]

        joint1_xcoords = joints[0::3]
        joint1_ycoords = joints[1::3]
        
        for i in range(17):
            joint1_heatmap[i] = generate_heatmaps([joint1_xcoords.astype(int)[i], joint1_ycoords.astype(int)[i]], bbox) 
        return joint1_heatmap

    def find_in_array(self, array, key):
        for i in range(len(array)):
            if key==array[i]["track_id"]:
                return i
        print("None is encountered")
        return None

    def find_track_ids(self, dict_keys):
        ids = []
        for i in range(len(dict_keys)):
            ids.append(dict_keys[i]["track_id"])
        return ids

    def extract_bounding_box(self, attribute):
        if "bbox" in attribute:
            bbox = attribute["bbox"]
        else: 
            bbox = None
        return bbox

    def find_center(self, frame):
        center_of_keypoints = np.average(frame["keypoints"][0::3]), np.average(frame["keypoints"][1::3])
        return center_of_keypoints         
    
    def normalize_points(self, frame, center):
        center = np.array(self.find_center(frame)) - np.array(center)
        frame["keypoints"][0:-1:3] = np.array(frame["keypoints"][0:-1:3]) - np.array(center[0])
        frame["keypoints"][1:-1:3] = np.array(frame["keypoints"][1:-1:3]) - np.array(center[1])
        return frame

    def testing_loop(self, gnn_model_path):
        model = torch.load(gnn_model_path)
        
    def save_heatmaps_as_npy(self, mode):
        backbone_model = HRNet()
        for k in range(len(self.history_file_joints)-4):

            frame1 = self.history_file_joints[k]
            frame2 = self.history_file_joints[k+1]
            frame3 = self.history_file_joints[k+2]
            frame4 = self.history_file_joints[k+3]

            human_ids4 = self.find_track_ids(frame4)            
            image_shape = cv2.imread(os.path.join("dataset/posetrack_data",self.history_files[k])).shape

            all_human_list = []

            for human_num in human_ids4:
                index4 = self.find_in_array(frame4, human_num)
                index3 = self.find_in_array(frame3, human_num)
                index2 = self.find_in_array(frame2, human_num)
                index1 = self.find_in_array(frame1, human_num)

                if index4 is None or index3 is None or index2 is None or index1 is None:
                    continue

                bbox4 = self.extract_bounding_box(frame4[index4])
                bbox3 = self.extract_bounding_box(frame3[index3])
                bbox2 = self.extract_bounding_box(frame2[index2])
                bbox1 = self.extract_bounding_box(frame1[index1])

                if bbox1 is None or bbox2 is None or bbox3 is None or bbox4 is None:
                    continue

                joint1_heatmap = self.get_hrnet_joint_embeddings(self.history_files[k], bbox1)
                joint2_heatmap = self.get_hrnet_joint_embeddings(self.history_files[k+1], bbox2)
                joint3_heatmap = self.get_hrnet_joint_embeddings(self.history_files[k+2], bbox3)
                joint4_heatmap = self.get_hrnet_joint_embeddings(self.history_files[k+3], bbox4)
                
                if joint1_heatmap is None or joint2_heatmap is None or joint3_heatmap is None or joint4_heatmap is None:
                    continue

                center_of_keypoints = self.find_center(frame3[index3])
                frame1[index1] = self.normalize_points(frame1[index1], center_of_keypoints)
                frame2[index2] = self.normalize_points(frame2[index2], center_of_keypoints)

                use_144_channels=False
                if use_144_channels:

                    image_path1 = os.path.join(self.image_folder_path, self.history_files[k])
                    image_path2 = os.path.join(self.image_folder_path, self.history_files[k+1])
                    image_path3 = os.path.join(self.image_folder_path, self.history_files[k+2])
                    image_path4 = os.path.join(self.image_folder_path, self.history_files[k+3])

                    joint1_heatmap = visual_feature_maps(backbone_model, image_path1, bbox1)
                    joint2_heatmap = visual_feature_maps(backbone_model, image_path2, bbox2)
                    joint3_heatmap = visual_feature_maps(backbone_model, image_path3, bbox3)
                    joint4_heatmap = visual_feature_maps(backbone_model, image_path4, bbox4)
                    
                    if len(joint1_heatmap)==0 or len(joint2_heatmap)==0 or len(joint3_heatmap)==0 or len(joint4_heatmap)==0:
                        continue

                    Jk1 = process_frame_joints(frame1[index1]["keypoints"], joint1_heatmap)
                    Jk2 = process_frame_joints(frame2[index2]["keypoints"], joint2_heatmap)
                    Jk3 = process_frame_joints(frame3[index3]["keypoints"], joint3_heatmap)

                human_attribute_list = {
                    "track_id":human_num,
                    "keypoints1":np.array(frame1[index1]["keypoints"]),
                    "keypoints2":np.array(frame2[index2]["keypoints"]),
                    "keypoints3":np.array(frame3[index3]["keypoints"]),
                    "keypoints4":np.array(frame4[index4]["keypoints"]),
                    "bbox_head1":np.array(frame1[index1]["bbox_head"]),
                    "bbox_head2":np.array(frame2[index2]["bbox_head"]),
                    "bbox_head3":np.array(frame3[index3]["bbox_head"]),
                    "bbox_head4":np.array(frame4[index4]["bbox_head"]),
                    "bbox1":bbox1,
                    "bbox2":bbox2,
                    "bbox3":bbox3,
                    "bbox4":bbox4,
                    "heatmaps1":joint1_heatmap,
                    "heatmaps2":joint2_heatmap,
                    "heatmaps3":joint3_heatmap,
                    "heatmaps4":joint4_heatmap,
                }
                
                all_human_list.append(human_attribute_list)
            images = [os.path.join("dataset/posetrack_data",self.history_files[k]), 
            os.path.join("dataset/posetrack_data",self.history_files[k+1]), 
            os.path.join("dataset/posetrack_data",self.history_files[k+2]),
            os.path.join("dataset/posetrack_data",self.history_files[k+3])]

            out_images_name = os.path.join("gnn_images", mode)
            joints_output_name = os.path.join("gnn_joints", mode)

            if all_human_list==[]:
                continue
            with open(os.path.join(out_images_name, str(k))+".npy", 'wb') as f:    
                pickle.dump(images, f)
            with open(os.path.join(joints_output_name, str(k))+".npy", 'wb') as f:    
                pickle.dump(np.array(all_human_list), f)

    def save_data(self):
        out_images_name = "images_npy"
        out_heatmaps_name = "heatmaps_npy"
        
        for i, _ in enumerate(tqdm(self.file_names)):
            read_image = cv2.imread(self.file_names[i])
            annotation_of_image = self.annotations[i]
            
            image, heatmap = self.prepare_data(read_image, annotation_of_image)

            with open(os.path.join(out_images_name, str(i))+".npy", 'wb') as f:    
                np.save(f, image)
            with open(os.path.join(out_heatmaps_name, str(i))+".npy", 'wb') as f:    
                np.save(f, heatmap)

    def prepare_data(self, read_image, annotation_of_image):
        joint_list = []

        it = iter(annotation_of_image["keypoints"])
        joint_counter=0
        for joint_x, joint_y, is_joint in zip(it, it, it):
            if is_joint==1:
                joint_list.append([joint_counter, joint_x, joint_y])
            joint_counter+=1

        x1 = int(annotation_of_image["bbox"][0])
        y1 = int(annotation_of_image["bbox"][1])
        x2 = int(x1+annotation_of_image["bbox"][2])
        y2 = int(y1+annotation_of_image["bbox"][3])
    
        cropped_image, resized_and_scaled_joints = crop_and_rescale_boxes_with_annots(read_image, [[x1,y1],[x2,y2]], joint_list)

        tmp_heat_maps = np.zeros((17, heatmap_width, heatmap_height))
        for joint_id, joint_x, joint_y in resized_and_scaled_joints:

            tmp_heat_maps[int(joint_id)] = generate_heatmaps((int(joint_x), int(joint_y))) 

        return cropped_image, tmp_heat_maps

    def __len__(self):
        return np.array(self.history_file_joints).shape[0]

    def __getitem__(self, idx):
        if True:
            return cv2.imread(os.path.join("dataset/posetrack_data",self.history_files[idx][0])), cv2.imread(os.path.join("dataset/posetrack_data",self.history_files[idx][1])), cv2.imread(os.path.join("dataset/posetrack_data",self.history_files[idx][2])), self.history_file_joints[idx]

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--is_train", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("-m", "--mode", help="select one of the following modes: train/val/test")
args = parser.parse_args()

if not args.is_train:
    dataset = PoseTrackDataset("dataset/posetrack_data", "dataset/posetrack_data/annotations", args.mode)
    dataset.save_heatmaps_as_npy(args.mode)
    exit()
else:
    dataset = GNNDataset("gnn_images", "gnn_joints", "gnn_heatmaps", args.mode)
    train_gnn(dataset)
    exit()
