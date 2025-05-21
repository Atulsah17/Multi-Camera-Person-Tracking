import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from scipy.spatial.distance import cosine
from collections import defaultdict, deque

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ReID model wrapper
class ReIDWrapper(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

# Load ReID model
reid_model = ReIDWrapper(models.resnet50(weights=ResNet50_Weights.DEFAULT)).eval().to(device)

# ReID transformations
reid_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Global ReID variables
global_features = {}
next_global_id = 1
feature_similarity_threshold = 0.4
within_video_threshold = 0.85
global_id_map = {}
track_features_buffer = defaultdict(lambda: deque(maxlen=30))

def extract_features(frame, bboxes, confidences):
    crops, valid_indices = [], []
    for idx, (bbox, conf) in enumerate(zip(bboxes, confidences)):
        if conf < 0.7:
            crops.append(None)
            continue
        x1, y1, x2, y2 = map(int, bbox)
        if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) < 100:
            crops.append(None)
            continue
        crop = frame[y1:y2, x1:x2]
        crops.append(crop)
        valid_indices.append(idx)

    valid_crops = [crop for crop in crops if crop is not None]
    if not valid_crops:
        return [None] * len(bboxes)

    batch = torch.stack([reid_transform(crop) for crop in valid_crops]).to(device)
    with torch.no_grad():
        features = reid_model(batch).cpu().numpy()
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    feat_list = [None] * len(bboxes)
    for i, idx in enumerate(valid_indices):
        feat_list[idx] = features[i].flatten()
    return feat_list

def match_person(agg_feature, cam_idx, local_id, existing_track_ids, used_gids):
    global next_global_id, global_features, global_id_map

    if cam_idx not in global_id_map:
        global_id_map[cam_idx] = defaultdict(int)

    if global_id_map[cam_idx][local_id] != 0:
        return global_id_map[cam_idx][local_id]

    if agg_feature is None:
        return -1

    agg_feature = agg_feature.astype(np.float64)
    for existing_local_id in existing_track_ids:
        if existing_local_id == local_id:
            continue
        existing_gid = global_id_map[cam_idx].get(existing_local_id, -1)
        if existing_gid == -1:
            continue
        mean_feat = np.mean(global_features[existing_gid], axis=0).astype(np.float64)
        sim = 1 - cosine(mean_feat, agg_feature)
        if sim > within_video_threshold:
            return -1

    best_gid, best_score = -1, 0
    for gid, feats_list in global_features.items():
        if gid in used_gids:
            continue
        mean_feat = np.mean(feats_list, axis=0).astype(np.float64)
        sim = 1 - cosine(mean_feat, agg_feature)
        if sim > best_score:
            best_score = sim
            if sim > feature_similarity_threshold:
                best_gid = gid

    if best_gid == -1:
        global_features[next_global_id] = [agg_feature]
        global_id_map[cam_idx][local_id] = next_global_id
        gid_assigned = next_global_id
        next_global_id += 1
        return gid_assigned
    else:
        global_features[best_gid].append(agg_feature)
        if len(global_features[best_gid]) > 15:
            global_features[best_gid] = global_features[best_gid][-15:]
        global_id_map[cam_idx][local_id] = best_gid
        return best_gid