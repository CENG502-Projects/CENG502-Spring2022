from tqdm import tqdm
import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn as detector_model
import json
from torch.utils.data import Dataset
import cv2
from os.path import exists
from models.hrnet import HRNet
import sys
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import json 
import natsort 
import argparse

device = torch.device(0)

width = 384
height = 288
heatmap_width = 96
heatmap_height = 72

def crop_boxes(img, box): 
    xmin = int(box[0])
    xmax = int(box[2])
    ymin = int(box[1])
    ymax = int(box[3])
    box = box.astype(int)

    tmp_image = img[ymin:ymax,xmin:xmax]

    if tmp_image.shape[0]==0 or tmp_image.shape[1]==0:
        print(img.shape, xmin,xmax,ymin,ymax)
        exit()
        return [], []

    tmp_image = cv2.resize(tmp_image, (height,width))
    cropped_image = tmp_image
            
    return cropped_image

def crop_all(img, boxes):
    cropped_images = []
    for i in range(len(boxes)):
        box_array_numpy = np.array(boxes[i])
        cropped_images.append(crop_boxes(img, box_array_numpy))
    return np.array(cropped_images)

def find_in_json(json_obj, attribute ,key):
    for i in range(len(json_obj)):
        if json_obj[i][attribute] == key:
            return json_obj[i]

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

def correct_old_order_representation(keypoints, bbox):

    keypoints[:,0] = keypoints[:, 0]*(bbox[2]-bbox[0])/heatmap_height
    keypoints[:,1] = keypoints[:, 1]*(bbox[3]-bbox[1])/heatmap_width

    bbox = np.array([bbox[0], bbox[1], 0])
    new_keys = np.zeros((17,3))
    new_keys[16] = keypoints[16]+bbox
    new_keys[15] = keypoints[15]+bbox
    new_keys[14] = keypoints[14]+bbox
    new_keys[13] = keypoints[13]+bbox
    new_keys[12] = keypoints[12]+bbox
    new_keys[11] = keypoints[11]+bbox
    new_keys[10] = keypoints[10]+bbox
    new_keys[9] = keypoints[9]+bbox
    new_keys[8] = keypoints[8]+bbox
    new_keys[7] = keypoints[7]+bbox
    new_keys[6] = keypoints[6]+bbox
    new_keys[5] = keypoints[5]+bbox
    new_keys[4] = [0,0,0]
    new_keys[3] = [0,0,0]    
    new_keys[2] = keypoints[2]+bbox
    new_keys[1] = keypoints[1]+bbox
    new_keys[0] = keypoints[0]+bbox
    return new_keys

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


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        
        x = x.view(batch_size, -1)

        h_1 = F.relu(self.input_fc(x))

        h_2 = F.relu(self.hidden_fc(h_1))

        y_pred = self.output_fc(h_2)

        return y_pred, h_2

feature_dimension = 500
gnn_feature_dimension = feature_dimension
num_hidden_channels = 500
gnn_out_channel_num = 500

position_MLP = None
visual_feature_MLP = None
type_embed = None


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

def process_frame_joints(joints, joint1_heatmap):
    joint1_heatmap = joint1_heatmap.reshape(17, -1)
    joints1 = torch.tensor(joints)#[0]
    joint1_xcoords = joints1[:,0::3]
    joint1_ycoords = joints1[:,1::3]
    joint1_is_visible = joints1[:,2::3]
    
    joint1_coord_pairs = torch.reshape(torch.stack((joint1_xcoords, joint1_ycoords, joint1_is_visible), axis=1),(17, 3))
    pos = position_MLP(joint1_coord_pairs.float())[0]
    vis = visual_feature_MLP(torch.from_numpy(joint1_heatmap).float())[0]
    embed = type_embed(torch.from_numpy(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])))
    concatenated = torch.cat((pos,embed,vis), axis=1)
    Jk = (pos+vis+embed)/3
    return Jk

def generate_human_dictionary(bboxes, track_ids, scores, joints):
    human_dict = {}
    human_dict["bbox"] = bboxes.tolist()
    human_dict["detection_score"] = scores.tolist()
    human_dict["track_id"] = track_ids
    human_dict["joints"] = joints.tolist()
    return human_dict

counter = 0

def generate_images_prediction_dictionary(is_labeled, filename, nframes, frame_id, vid_id):
    global counter
    image_dict = {}
    if is_labeled==False:
        image_dict["frame_id"]=counter
        image_dict["id"]=counter
        image_dict["vid_id"]=counter
        counter+=1
    else:
        image_dict["frame_id"]=frame_id
        image_dict["id"]=frame_id
        image_dict["vid_id"]=vid_id
    image_dict["is_labeled"]=is_labeled
    image_dict["file_name"]=filename
    image_dict["nframes"]=nframes
    return image_dict

def generate_annotations_prediction_dictionary(is_labeled, keypoints, track_id, image_id, bbox):
    anno_dict = {}
    track_string = ""
    if is_labeled==False:
        id = counter
        image_id = counter
    else:
        if track_id>9:
            track_string=str(image_id)+str(track_id)
        else:
            track_string=str(image_id)+"0"+str(track_id)
        id = int(track_string)
    anno_dict["id"] = id
    anno_dict["keypoints"] = keypoints
    anno_dict["track_id"] = track_id
    anno_dict["image_id"] = image_id
    anno_dict["bbox"] = bbox
    anno_dict["category_id"] = 1
    anno_dict["bbox_head"] = bbox
    anno_dict["scores"] = []
    return anno_dict

def generate_prediction_dictionary(humans, filename, root_dictionary, is_labeled_true):
    correct_dict = []
    annorect = []
    for human in humans:
        point_list = []
        for joint_id, joint in enumerate(human["joints"]):
            if joint[2]==0:
                joint[0] = 0
                joint[1]
                pass
            point_list.append({
                "id":[joint_id],
                "x":[joint[0]],
                "y":[joint[1]],
                "score":[1.0]
                })
        point_dict = {"point": point_list}
        annorect.append({
            "annopoints": [point_dict],        
            "x1":[human["bbox"][0]],
            "x2":[human["bbox"][2]],
            "y1":[human["bbox"][1]],
            "y2":[human["bbox"][3]],
            "track_id":[human["track_id"]],
            "score":[human["detection_score"]]
            })
        if is_labeled_true:
            correct_dict.append({
                "annorect": annorect, 
                "image": [{"name":filename}]
                })
        else:
            correct_dict.append({
                "annorect": {},
                "image": [{"name":filename}]
                })
            
    correct_dict
    return correct_dict

class PoseTrackTestDataset(Dataset):
    def __init__(self, image_folder_path, all_annotation_paths, mode, transform=None):
        self.categories = {}
        self.image_folder_path = image_folder_path
        self.annotation_files = []
        self.annotation_images_names = []
        self.annotation_file_name = []
        self.image_exists_in_all = []
        self.image_labeled_in_all = []
        self.video_ids_in_all = []
        self.frame_ids_in_all = []
        self.default_value = self.initialize_default()
        flag = 1
        annotation_file_paths = os.listdir(os.path.join(all_annotation_paths, mode))
        for path in tqdm(annotation_file_paths):
            video_id = []
            frame_id = []
            image_exists = []
            does_image_labeled = []
            self.annotation_file_name.append(path[:-5])
            images_path = []
            images_names = []
            
            path_to_file = os.path.join(all_annotation_paths, mode, path)
            file_obj = open(path_to_file)
            annotation_file = json.load(file_obj)
            for i in range(0,len(annotation_file["images"])):
                self.categories = annotation_file["categories"]
                images = annotation_file["images"]             
                video_id.append(images[i]["vid_id"])     
                frame_id.append(images[i]["frame_id"])   
                image_labeled1 = images[i]["is_labeled"]
                images_names.append(images[i]["file_name"])
                
                image1_abs_path = os.path.join("dataset/posetrack_data", images[i-3]["file_name"])
                
                if i<3:
                    does_image_labeled.append(False)
                    images_path.append([None, None, None, None])
                    image_exists.append(0)
                else:
                    image_labeled4 = images[i]["is_labeled"]
                    
                    image2_abs_path = os.path.join("dataset/posetrack_data", images[i-2]["file_name"])
                    image3_abs_path = os.path.join("dataset/posetrack_data", images[i-1]["file_name"])
                    image4_abs_path = os.path.join("dataset/posetrack_data", images[i]["file_name"])
                    images_path.append([image1_abs_path, image2_abs_path, image3_abs_path, image4_abs_path])

                    label_image = image_labeled4==True
                    if not exists(image1_abs_path) or not label_image:
                        image_exists.append(0)
                    else:
                        image_exists.append(1)
                        flag=0
            self.video_ids_in_all.append(video_id)
            self.frame_ids_in_all.append(frame_id)
            self.image_labeled_in_all.append(does_image_labeled)
            self.image_exists_in_all.append(image_exists)
            self.annotation_files.append(images_path)
            self.annotation_images_names.append(images_names)

    def initialize_default(self):
        empty_heatmaps = np.zeros((1, 17,heatmap_width, heatmap_height))
        default_attributes  = {
            "frame_id" : 0,
            "vid_id" : 0,
            "track_ids1": [0],
            "track_ids2": [0],
            "track_ids3": [0],
            "track_ids4": [0],
            "human_bounding_boxes1": [np.array([0,0,0,0])],
            "human_bounding_boxes2": [np.array([0,0,0,0])],
            "human_bounding_boxes3": [np.array([0,0,0,0])],
            "human_bounding_boxes4": [np.array([0,0,0,0])],
            "hrnet_heatmaps1": [empty_heatmaps],
            "hrnet_heatmaps2": [empty_heatmaps],
            "hrnet_heatmaps3": [empty_heatmaps],
            "hrnet_heatmaps4": [empty_heatmaps],
            "detection_scores1":[np.array((0))],
            "detection_scores2":[np.array((0))],
            "detection_scores3":[np.array((0))],
            "detection_scores4":[np.array((0))],
            "label_frame":False
        }
        return default_attributes

    def extract_humans(self, dictionary_output):
        boxes = []
        scores = []
        for i in range(len(dictionary_output["labels"])):
            if dictionary_output["labels"][i]==1 and dictionary_output["scores"][i]>0.5:
                boxes.append(dictionary_output["boxes"][i])                
                scores.append(dictionary_output["scores"][i])   
        return boxes, scores

    def get_image(self, image_numpy):
        image_torch = torch.from_numpy(image_numpy)
        image_permuted = torch.permute(image_torch, (2,0,1))
        image_unsqueezed = image_permuted.unsqueeze(0).float()
        return image_unsqueezed

    def get_all_centers(self, human_boxes):
        centers = []
        for i in range(len(human_boxes)):
            centers.append(self.get_center(human_boxes[i]))
        return centers

    def get_center(self, human_boxes):
        return [(human_boxes[0]+human_boxes[2])/2, (human_boxes[1]+human_boxes[3])/2]

    def compare_distance_to_center(self, center_of_box, current_center):
        return np.linalg.norm(np.array(center_of_box) - np.array(current_center))

    def assign_track_ids(self, human_boxes, current_track_ids, current_centers):
        track_ids = []
        if current_track_ids==[]:
            for track_id in range(len(human_boxes)):
                track_ids.append(track_id)
        else:
            for boxes_indices in range(len(human_boxes)):
                min_distance = sys.float_info.max
                selected_indice = -1
                for center_indices in range(len(current_centers)):
                    center_of_box = self.get_center(human_boxes[boxes_indices])        
                    distance = self.compare_distance_to_center(center_of_box, current_centers[center_indices])
                    if distance < min_distance:
                        min_distance = distance
                        selected_indice = center_indices
                track_ids.append(selected_indice)                
        return track_ids

    def is_in_all_frames(self, key, current_track_ids, track_ids1, track_ids2, track_ids3):
        if key in track_ids1 and key in track_ids2 and key in track_ids3 and key in current_track_ids:
            return True
        return False

    def get_persisting_detections(self, current_track_ids, track_ids1, track_ids2, track_ids3):
        track_id_indices1 = []
        track_id_indices2 = []
        track_id_indices3 = []

        for i in range(len(current_track_ids)):
            if current_track_ids[i] in track_ids1 and current_track_ids[i] in track_ids2 and current_track_ids[i] in track_ids3:
                track_id_indices1.append(track_ids1.index(current_track_ids[i]))
                track_id_indices2.append(track_ids2.index(current_track_ids[i]))
                track_id_indices3.append(track_ids3.index(current_track_ids[i]))
        return track_id_indices1, track_id_indices2, track_id_indices3, len(track_id_indices1)

    def get_nth_data(self, data, indices):
        new_data = []
        for i in range(len(indices)):
            new_data.append(data[indices[i]])
        return new_data

    def convert_tensor_to_numpy(self, tensor_array):
        new_array = []
        for i in range(len(tensor_array)):
            new_array.append(tensor_array[i].cpu().detach().numpy())
        return new_array

    def get_human_indice_by_track_id(self, key, track_ids):
        for ith_index, track_id in enumerate(track_ids): 
            if key==track_id:
                return ith_index
        return -1

    def convert_numpy_to_tensor(self, numpy_image):
        global device
        torch_image = torch.from_numpy(numpy_image)
        permuted_image = torch.permute(torch_image, (2, 0, 1)).float()
        unsqueezed_image = permuted_image.unsqueeze(0)
        return unsqueezed_image.to(device)

    def process_heatmaps(self, pose_model, image, detection_boxes):
        heatmap_list = []
        for i in range(len(detection_boxes)):
            bbox = detection_boxes[i].astype(int)
            cropped_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            cropped_image = cv2.resize(cropped_image, (height, width))
            heatmap = pose_model(self.convert_numpy_to_tensor(cropped_image)).cpu().detach().numpy()
            heatmap_list.append(heatmap)
        return np.array(heatmap_list)

    def save_data_as_npy(self, mode):
        device = torch.device(0)
        detection = detector_model(pretrained=True).to(device)
        detection.eval()
        pose_model = HRNet(32, 17, 0.1)      
        pose_model.load_state_dict(torch.load("pose_hrnet_w32_384x288.pth"))
        joints_output_name = os.path.join("detection_preprocessed_inputs", mode)
        pose_model = pose_model.to(device)
        for anno_idx, _ in enumerate(self.annotation_file_name):
            anno_file_output_path = os.path.join(joints_output_name, self.annotation_file_name[anno_idx])
            os.makedirs(anno_file_output_path, exist_ok=True)
            annotation = self.annotation_files[anno_idx]
            for i in range(len(annotation)):
                human_attribute_list = None
                if self.image_exists_in_all[anno_idx][i]==0:
                    human_attribute_list = self.default_value
                    human_attribute_list["filename"] = self.annotation_images_names[anno_idx][i]
                else:
                    image1_numpy = cv2.imread(annotation[i][0])/255
                    image2_numpy = cv2.imread(annotation[i][1])/255
                    image3_numpy = cv2.imread(annotation[i][2])/255
                    image4_numpy = cv2.imread(annotation[i][3])/255
                    
                    image1 = self.get_image(image1_numpy)
                    image2 = self.get_image(image2_numpy)
                    image3 = self.get_image(image3_numpy)
                    image4 = self.get_image(image4_numpy)
                    
                    detection_output1 = detection(image1.to(device))
                    detection_output2 = detection(image2.to(device))
                    detection_output3 = detection(image3.to(device))
                    detection_output4 = detection(image4.to(device))
                    
                    human_boxes1, human_scores1 = self.extract_humans(detection_output1[0])
                    human_boxes2, human_scores2 = self.extract_humans(detection_output2[0])
                    human_boxes3, human_scores3 = self.extract_humans(detection_output3[0])
                    human_boxes4, human_scores4 = self.extract_humans(detection_output4[0])
                    
                    human_boxes1 = self.convert_tensor_to_numpy(human_boxes1)
                    human_boxes2 = self.convert_tensor_to_numpy(human_boxes2)
                    human_boxes3 = self.convert_tensor_to_numpy(human_boxes3)
                    human_boxes4 = self.convert_tensor_to_numpy(human_boxes4)
                    human_scores1 = self.convert_tensor_to_numpy(human_scores1)
                    human_scores2 = self.convert_tensor_to_numpy(human_scores2)
                    human_scores3 = self.convert_tensor_to_numpy(human_scores3)
                    human_scores4 = self.convert_tensor_to_numpy(human_scores4)
                    
                    current_track_ids = self.assign_track_ids(human_boxes4, [], None)
                    current_centers = self.get_all_centers(human_boxes4)
                    
                    track_ids1 = self.assign_track_ids(human_boxes1, current_track_ids, current_centers)
                    track_ids2 = self.assign_track_ids(human_boxes2, current_track_ids, current_centers)
                    track_ids3 = self.assign_track_ids(human_boxes3, current_track_ids, current_centers)
                    track_ids4 = current_track_ids
                    
                    hrnet_heatmaps1 = self.process_heatmaps(pose_model, image1_numpy, human_boxes1)
                    hrnet_heatmaps2 = self.process_heatmaps(pose_model, image2_numpy, human_boxes2)
                    hrnet_heatmaps3 = self.process_heatmaps(pose_model, image3_numpy, human_boxes3)
                    hrnet_heatmaps4 = self.process_heatmaps(pose_model, image4_numpy, human_boxes4)
                    
                    human_attribute_list = {
                        "frame_id": self.frame_ids_in_all[anno_idx][i],
                        "vid_id": self.video_ids_in_all[anno_idx][i],
                        "track_ids1": track_ids1,
                        "track_ids2": track_ids2,
                        "track_ids3": track_ids3,
                        "track_ids4": track_ids4,
                        "human_bounding_boxes1": human_boxes1,
                        "human_bounding_boxes2": human_boxes2,
                        "human_bounding_boxes3": human_boxes3,
                        "human_bounding_boxes4": human_boxes4,
                        "hrnet_heatmaps1": hrnet_heatmaps1,
                        "hrnet_heatmaps2": hrnet_heatmaps2,
                        "hrnet_heatmaps3": hrnet_heatmaps3,
                        "hrnet_heatmaps4": hrnet_heatmaps4,
                        "detection_scores1":human_scores1,
                        "detection_scores2":human_scores2,
                        "detection_scores3":human_scores3,
                        "detection_scores4":human_scores4,
                        "filename":self.annotation_images_names[anno_idx][i],
                        "label_frame":True
                    }
                with open(os.path.join(anno_file_output_path , str(i))+".npy", 'wb') as f:    
                    pickle.dump(np.array(human_attribute_list), f)

    def merge_output_annotations(self, gnn_model_output, hrnet_output):
        merged_outputs = []
        for joint_idx in range(len(hrnet_output)):
            if hrnet_output[joint_idx][2]==0 and gnn_model_output[joint_idx][2]>0.5:
                merged_outputs.append(np.array(gnn_model_output[joint_idx]))
            else:
                merged_outputs.append(np.array(hrnet_output[joint_idx]))
        return np.array(merged_outputs)

    def generate_joints(self, heatmap):
        joints = np.zeros((17, 3)).astype(np.float32)
        for j in range(heatmap.shape[1]):
            joints[j][:2] = np.unravel_index(heatmap[0][j].argmax(), heatmap[0][j].shape)
            if (joints[j][:2] == [0,0]).all():
                joints[j][2] = 0
            else:
                joints[j][2] = 1
        tmp = joints[:, 0].copy()
        joints[:, 0] = joints[:, 1]
        joints[:, 1] = tmp
        
        return joints
    
    def correct_bounding_boxes(self, input_dict):
        for i in range(len(input_dict["human_bounding_boxes4"])):
            tmp_x = input_dict["human_bounding_boxes4"][i][0]
            tmp_width = input_dict["human_bounding_boxes4"][i][2]
            
            input_dict["human_bounding_boxes4"][i][0] = input_dict["human_bounding_boxes4"][i][1]
            input_dict["human_bounding_boxes4"][i][1] = tmp_x
            input_dict["human_bounding_boxes4"][i][2] = input_dict["human_bounding_boxes4"][i][3]
            input_dict["human_bounding_boxes4"][i][3] = tmp_width
        return input_dict

    def convert_to_width_height_bbox(self, bbox):
        return [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]

    def test_loop(self, folder_path):
        global position_MLP, visual_feature_MLP, type_embed
        device = torch.device(0)
        can_generate_only_one_label = False

        gnn_related_models = torch.load("gnn_models/1.pt")
        pred_MLP = gnn_related_models["pred_MLP"].to(device)            
        position_MLP = gnn_related_models["position_MLP"]   
        visual_feature_MLP = gnn_related_models["visual_feature_MLP"]  
        type_embed = gnn_related_models["type_embed"]            
        gnn_model = gnn_related_models["gnn_model"]

        folders = os.listdir(folder_path)

        global_adjaceny1 = generate_global_edges(1)
        global_adjaceny2 = generate_global_edges(2)
        self_adjaceny1 = create_edge_representations(0)
        self_adjaceny2 = create_edge_representations(1)
        self_adjaceny3 = create_edge_representations(2)

        json_result_path = "results"

        need_correction = True
        for folder in folders:
            result_dict = {}
            result_dict["images"] = []
            result_dict["annotations"] = []
            absolute_folder_path = os.path.join(folder_path, folder)
            files = natsort.natsorted(os.listdir(absolute_folder_path))
            
            open(os.path.join(json_result_path, folder)+".json", 'w').close()
            for i in range(len(files)):    
                absolute_saved_npy_path = os.path.join(absolute_folder_path, files[i])
                with open(absolute_saved_npy_path, "rb") as input_file:
                    input_dictionary = pickle.load(input_file).item()
                num_humans = input_dictionary["track_ids4"]
                all_humans = []
                is_labeled_true = True
                if input_dictionary["detection_scores4"][0]==0:
                    is_labeled_true = False
                    result_dict["images"].append(generate_images_prediction_dictionary(is_labeled_true, input_dictionary["filename"], len(files), input_dictionary["frame_id"], input_dictionary["vid_id"]))
                    continue
                if input_dictionary["label_frame"]==False or can_generate_only_one_label==True:
                    human_joint = self.generate_joints(input_dictionary["hrnet_heatmaps4"][0])
                    if need_correction:
                        human_joint = correct_old_order_representation(human_joint, input_dictionary["human_bounding_boxes4"][0])
                        human_joint = human_joint.flatten()
                    human = generate_human_dictionary(input_dictionary["human_bounding_boxes4"][0], input_dictionary["track_ids4"][0], input_dictionary["detection_scores4"][0], 
                    human_joint)
                    
                    human["bbox"] = self.convert_to_width_height_bbox(human["bbox"])
                    all_humans.append(human) 
                    is_labeled_true = False       
                                 
                    result_dict["annotations"].append(generate_annotations_prediction_dictionary(is_labeled_true, human["joints"], human["track_id"], 0, human["bbox"]))
                else:
                    can_generate_only_one_label = False
                    for human_idx in num_humans:
                        if input_dictionary["detection_scores4"][human_idx]<0.9:
                            continue
                        human_indice1 = self.get_human_indice_by_track_id(human_idx, input_dictionary["track_ids1"])
                        human_indice2 = self.get_human_indice_by_track_id(human_idx, input_dictionary["track_ids2"])
                        human_indice3 = self.get_human_indice_by_track_id(human_idx, input_dictionary["track_ids3"])
                        human_indice4 = self.get_human_indice_by_track_id(human_idx, input_dictionary["track_ids4"])

                        track_ids1 = input_dictionary["track_ids1"]
                        track_ids2 = input_dictionary["track_ids2"]
                        track_ids3 = input_dictionary["track_ids3"]
                        track_ids4 = input_dictionary["track_ids4"]

                        if not self.is_in_all_frames(input_dictionary["track_ids4"][human_indice4], track_ids4, track_ids1, track_ids2, track_ids3):
                            continue
                        
                        joints1 = self.generate_joints(input_dictionary["hrnet_heatmaps1"][human_indice1]*255)
                        joints2 = self.generate_joints(input_dictionary["hrnet_heatmaps2"][human_indice2]*255)
                        joints3 = self.generate_joints(input_dictionary["hrnet_heatmaps3"][human_indice3]*255)
                        joints4 = self.generate_joints(input_dictionary["hrnet_heatmaps4"][human_indice4]*255)
                        image_name = "dataset/posetrack_data/"+input_dictionary["filename"]

                        use_gnn=True
                        if use_gnn:
                            Jk1 = process_frame_joints(joints1, input_dictionary["hrnet_heatmaps1"][human_indice1]*255)
                            Jk2 = process_frame_joints(joints2, input_dictionary["hrnet_heatmaps2"][human_indice2]*255)
                            Jk3 = process_frame_joints(joints3, input_dictionary["hrnet_heatmaps3"][human_indice3]*255)
                            
                            data = create_dictionary_data([Jk1,Jk2,Jk3], [global_adjaceny1, global_adjaceny2], [self_adjaceny1, self_adjaceny2, self_adjaceny3])
                            gnn_outputs = gnn_model(data.x_dict, data.edge_index_dict)[('frame3', 'self_f', 'frame3')].to(device)
                            gnn_outputs = pred_MLP(gnn_outputs)[0].cpu().detach().numpy()

                            merged_joints = self.merge_output_annotations(gnn_outputs, joints4)

                            merged_joints = gnn_outputs
                        else:
                            merged_joints = joints4
                        if need_correction:
                            merged_joints = correct_old_order_representation(merged_joints, input_dictionary["human_bounding_boxes4"][human_indice4])
                            merged_joints = merged_joints.flatten()
                        human = generate_human_dictionary(input_dictionary["human_bounding_boxes4"][human_indice4], track_ids4[human_indice4], input_dictionary["detection_scores4"][human_indice4], merged_joints)
                
                        human["bbox"] = self.convert_to_width_height_bbox(human["bbox"])
                        all_humans.append(human)    
                        result_dict["annotations"].append(generate_annotations_prediction_dictionary(is_labeled_true, human["joints"], human["track_id"], input_dictionary["frame_id"], human["bbox"]))
                result_dict["images"].append(generate_images_prediction_dictionary(is_labeled_true, input_dictionary["filename"], len(files), input_dictionary["frame_id"], input_dictionary["vid_id"]))
                result_dict["categories"] = self.categories
                                                
            json_result = json.dumps(result_dict)
            with open(os.path.join(json_result_path, folder)+".json", 'w') as outfile:
                outfile.write(json_result)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", help="select one of the following modes: train/val/test")
args = parser.parse_args()

dataset = PoseTrackTestDataset("dataset/posetrack_data", "dataset/posetrack_data/annotations", args.mode)
dataset.save_data_as_npy(args.mode)
dataset.test_loop(os.path.join("detection_preprocessed_inputs", args.mode))
