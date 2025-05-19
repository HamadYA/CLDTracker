import torchvision.transforms as transforms
import torch
import lib.models.layers.clip.clip as clip
from PIL import Image
from torch import nn
import json, os, sys

# TTFUM
from collections import deque

current_dir = os.path.dirname(os.path.abspath(__file__))
cldtracker_path = os.path.realpath(os.path.join(current_dir, "../../.."))
if cldtracker_path not in sys.path:
    sys.path.append(cldtracker_path)

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    BICUBIC = Image.BICUBIC
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=BICUBIC),
        transforms.CenterCrop(n_px),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def load_json(filename: str):
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as fp:
        return json.load(fp)


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.fc(x)
        return x


class Category_embedding(nn.Module):

    def __init__(self):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load('ViT-B/32', self.device)
        
        self.convert_vector = torch.nn.Linear(512, 768).to(self.device)
        self.softmax = nn.Softmax(dim=-1)

        ###############
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype
        self.adapter = Adapter(512, 4).to(self.dtype).to(self.device)
        self.ratio = 0.2
        self.logit_scaling = self.logit_scale.exp()
        ###############

        # COCO Labels
        self.label = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                      'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                      'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                      'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                      'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                      'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                      'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                      'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                      'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                      'teddy bear', 'hair drier', 'toothbrush']
        
        sequence_name_path = 'sequence_name.txt'

        if os.path.exists(sequence_name_path):

            print("CLDTracker")

            # TrackingNet
            trackingnet_dict_path = os.path.join(cldtracker_path, 'comprehensive_bag_of_textual_descriptions/final_bag/trackingnet.json')
            trackingnet_json = load_json(trackingnet_dict_path)

            # LaSOT
            lasot_dict_path = os.path.join(cldtracker_path, 'comprehensive_bag_of_textual_descriptions/final_bag/lasot.json')
            lasot_json = load_json(lasot_dict_path)

            # LaSOT EXT
            lasot_ext_dict_path = os.path.join(cldtracker_path, 'comprehensive_bag_of_textual_descriptions/final_bag/lasot_ext.json')
            lasot_ext_json = load_json(lasot_ext_dict_path)

            # GOT-10k
            got10k_dict_path = os.path.join(cldtracker_path, 'comprehensive_bag_of_textual_descriptions/final_bag/got10k.json')
            got10k_json = load_json(got10k_dict_path)

            # COCO
            coco_dict_path = os.path.join(cldtracker_path, 'comprehensive_bag_of_textual_descriptions/final_bag/coco.json')
            coco_json = load_json(coco_dict_path)

            # OTB
            otb_dict_path = os.path.join(cldtracker_path, 'comprehensive_bag_of_textual_descriptions/final_bag/otb.json')
            otb_json = load_json(otb_dict_path)

            # TNL2K
            tnl2k_dict_path = os.path.join(cldtracker_path, 'comprehensive_bag_of_textual_descriptions/final_bag/tnl2k.json')
            tnl2k_json = load_json(tnl2k_dict_path)

            # D-PTUAC
            dptuac_dict_path = os.path.join(cldtracker_path, 'comprehensive_bag_of_textual_descriptions/final_bag/dptuac.json')
            dptuac_json = load_json(dptuac_dict_path)
            

            with open(sequence_name_path, 'r') as file:
                sequence_name = file.read()
                
            descriptions = {sequence_name: lasot_json[sequence_name]}
            for key in descriptions:
                bag = descriptions[key]
            
            print(bag)
            self.text_inputs = torch.cat([clip.tokenize(f"a photo of a {v}.") for v in bag]).to(self.device)

        else:
            print("CLIP COCO Lables")
            self.text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}.") for c in self.label]).to(self.device)


        with torch.no_grad():
            self.text_features = self.clip_model.encode_text(self.text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
            self.text = self.convert_vector(self.text_features.float()).softmax(dim=-1)


        self.tem_image_features = None
        self.tem_similarity = None
        self.indices = None
        window_size = 5
        self.sem_ttfum = deque(maxlen=window_size)


    def forward(self, template, search):
        # image-encoder
        if self.training is True or type(template)==torch.Tensor:
            batch_size = len(template)
        else:
            batch_size = 1
        class_des = torch.Tensor().cuda()
        search_des = torch.Tensor().cuda()
        attention_des = torch.Tensor().cuda()

        for i in range(batch_size):
            if self.training:
                ToPILImage = transforms.ToPILImage()
                tem_img_PIL = ToPILImage(template[i])
                tem_image_input = self.preprocess(tem_img_PIL).unsqueeze(0).to(self.device)
                sea_img_PIL = ToPILImage(search[i])
                sea_image_input = self.preprocess(sea_img_PIL).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    tem_image_features = self.clip_model.encode_image(tem_image_input)
                    x = self.adapter(tem_image_features)
                    tem_image_features = self.ratio * x + (1 - self.ratio) * tem_image_features
                    tem_image_features /= tem_image_features.norm(dim=-1, keepdim=True)
                    tem_logits = self.logit_scaling * tem_image_features @ self.text_features.t()
                    tem_similarity = (100.0 * tem_image_features @ self.text_features.T.half()).softmax(dim=-1)
                    tem_values, tem_indices = tem_similarity[0].topk(1)
                    tem_class_label = self.text[tem_indices]
                    
                    sea_image_features = self.clip_model.encode_image(sea_image_input)
                    x = self.adapter(sea_image_features)
                    sea_image_features = self.ratio * x + (1 - self.ratio) * sea_image_features
                    sea_image_features /= sea_image_features.norm(dim=-1, keepdim=True)
                    sem_logits = self.logit_scaling * sea_image_features @ self.text_features.t()
                    sem_similarity = (100.0 * sea_image_features @ self.text_features.T.half()).softmax(dim=-1)
                    sem_values, sem_indices = sem_similarity[0].topk(1)
                    sem_class_label = self.text[sem_indices]

                    class_attention = -torch.norm(tem_logits - sem_logits)

                    attention = self.softmax(torch.Tensor([class_attention])).unsqueeze(0)

            else:
                if self.tem_image_features is None:
                    with torch.no_grad():
                        tem_image_input = self.preprocess(template).unsqueeze(0).to(self.device)
                        self.tem_image_features = self.clip_model.encode_image(tem_image_input)
                        x = self.adapter(self.tem_image_features)
                        self.tem_image_features = self.ratio * x + (1 - self.ratio) * self.tem_image_features
                        self.tem_image_features /= self.tem_image_features.norm(dim=-1, keepdim=True)
                        self.tem_logits = self.logit_scaling * self.tem_image_features @ self.text_features.t()
                        self.tem_similarity = (100.0 * self.tem_image_features @ self.text_features.T.half()).softmax(dim=-1)
                        self.tem_values, self.tem_indices = self.tem_similarity[0].topk(1)
            

                sea_image_input = self.preprocess(search).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    sea_image_features = self.clip_model.encode_image(sea_image_input)
                    x = self.adapter(sea_image_features)
                    sea_image_features = self.ratio * x + (1 - self.ratio) * sea_image_features
                    sea_image_features /= sea_image_features.norm(dim=-1, keepdim=True)
                    sem_logits = self.logit_scaling * sea_image_features @ self.text_features.t()
                    sem_similarity = (100.0 * sea_image_features @ self.text_features.T.half()).softmax(dim=-1)
                    sem_values, sem_indices = sem_similarity[0].topk(1)
                    sem_class_label = self.text[sem_indices]

                    # TTFUM
                    self.sem_ttfum.append(sem_logits)
                    sem_class_ttfum = list(self.sem_ttfum)
                    class_attention = -torch.norm(self.tem_logits - torch.stack(sem_class_ttfum).mean(dim=0))
                    attention = self.softmax(torch.Tensor([class_attention])).unsqueeze(0)

                    tem_class_label = self.text[self.tem_indices]


            class_des = torch.cat([class_des, tem_class_label], dim=0)
            search_des = torch.cat([search_des, sem_class_label], dim=0)
            attention_des = torch.cat([attention_des, attention.cuda()], dim=0)

            return class_des.resize(batch_size, 1, 768, 1 ,1 ), search_des.resize(batch_size, 1, 768, 1 ,1 ), attention_des
        
