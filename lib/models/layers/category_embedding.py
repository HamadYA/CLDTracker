import torchvision.transforms as transforms
import torch
import lib.models.layers.clip.clip as clip
from PIL import Image
from torch import nn
import json, os, sys
from collections import deque

# Setup path
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
        return self.fc(x)

class Category_embedding(nn.Module):
    def __init__(self, training=False):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load('ViT-B/32', self.device)

        self.convert_vector = torch.nn.Linear(512, 768).to(self.device)
        self.softmax = nn.Softmax(dim=-1)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype
        self.adapter = Adapter(512, 4).to(self.dtype).to(self.device)
        self.ratio = 0.2
        self.logit_scaling = self.logit_scale.exp()

        self.training = training

        sequence_name_path = 'sequence_name.txt'
        dataset_name_file = 'dataset_name.txt'

        if training:
            print("CLDTracker: train mode")
            with open(sequence_name_path, 'r') as f:
                sequence_name = [line.strip() for line in f.readlines()]
            with open(dataset_name_file, 'r') as f:
                dataset_name = [line.strip() for line in f.readlines()]

            json_paths = {
                'trackingnet': load_json(os.path.join(cldtracker_path, 'comprehensive_bag_of_textual_descriptions/final_bag/trackingnet.json')),
                'lasot': load_json(os.path.join(cldtracker_path, 'comprehensive_bag_of_textual_descriptions/final_bag/lasot.json')),
                'got10k': load_json(os.path.join(cldtracker_path, 'comprehensive_bag_of_textual_descriptions/final_bag/got10k.json')),
                'coco': load_json(os.path.join(cldtracker_path, 'comprehensive_bag_of_textual_descriptions/final_bag/coco.json')),
            }

            final_list = []
            for seq, ds in zip(sequence_name, dataset_name):
                json_data = json_paths.get(ds, {})
                if seq in json_data:
                    final_list.append(json_data[seq])

            max_length = max(len(sublist) for sublist in final_list)
            extended = [sublist * (max_length // len(sublist)) + sublist[:max_length % len(sublist)] for sublist in final_list]
            tokenized = [torch.cat([clip.tokenize(f"a photo of a {v}.") for v in sublist]) for sublist in extended]
            self.text_inputs = torch.stack(tokenized).to(self.device)

            with torch.no_grad():
                self.text_features = torch.stack([self.clip_model.encode_text(v) for v in self.text_inputs])

        else:
            print("CLDTracker: test mode")
            with open(sequence_name_path, 'r') as file:
                sequence_name = file.read().strip()

            lasot_path = os.path.join(cldtracker_path, 'comprehensive_bag_of_textual_descriptions/final_bag/lasot.json')
            lasot_json = load_json(lasot_path)
            descriptions = {sequence_name: lasot_json[sequence_name]}
            for key in descriptions:
                bag = descriptions[key]

            self.text_inputs = torch.cat([clip.tokenize(f"a photo of a {v}.") for v in bag]).to(self.device)
            with torch.no_grad():
                self.text_features = self.clip_model.encode_text(self.text_inputs)
        # else:
            # raise RuntimeError("Expected sequence_name.txt (and optionally dataset_name.txt) for mode selection")

        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        self.text = self.convert_vector(self.text_features.float()).softmax(dim=-1)
        self.tem_image_features = None
        self.tem_similarity = None
        self.tem_indices = None
        self.sem_ttfum = deque(maxlen=5)

    def forward(self, template, search):
        batch_size = len(template) if self.training or isinstance(template, torch.Tensor) else 1
        class_des = torch.Tensor().cuda()
        search_des = torch.Tensor().cuda()
        attention_des = torch.Tensor().cuda()

        for i in range(batch_size):
            if self.training:
                to_pil = transforms.ToPILImage()
                tem_img = self.preprocess(to_pil(template[i])).unsqueeze(0).to(self.device)
                sea_img = self.preprocess(to_pil(search[i])).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    tem_feat = self.clip_model.encode_image(tem_img)
                    tem_feat = self.ratio * self.adapter(tem_feat) + (1 - self.ratio) * tem_feat
                    tem_feat /= tem_feat.norm(dim=-1, keepdim=True)
                    tem_logits = self.logit_scaling * tem_feat @ self.text_features[i].t()
                    tem_sim = (100.0 * tem_feat @ self.text_features[i].T.half()).softmax(dim=-1)
                    _, tem_idx = tem_sim[0].topk(1)
                    tem_cls = self.text[i][tem_idx]

                    sea_feat = self.clip_model.encode_image(sea_img)
                    sea_feat = self.ratio * self.adapter(sea_feat) + (1 - self.ratio) * sea_feat
                    sea_feat /= sea_feat.norm(dim=-1, keepdim=True)
                    sea_logits = self.logit_scaling * sea_feat @ self.text_features[i].t()
                    sea_sim = (100.0 * sea_feat @ self.text_features[i].T.half()).softmax(dim=-1)
                    _, sea_idx = sea_sim[0].topk(1)
                    sea_cls = self.text[i][sea_idx]

                    attention = self.softmax(torch.Tensor([-torch.norm(tem_logits - sea_logits)])).unsqueeze(0)
            else:
                if self.tem_image_features is None:
                    tem_img = self.preprocess(template if batch_size == 1 else transforms.ToPILImage()(template[i])).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        self.tem_image_features = self.clip_model.encode_image(tem_img)
                        self.tem_image_features = self.ratio * self.adapter(self.tem_image_features) + (1 - self.ratio) * self.tem_image_features
                        self.tem_image_features /= self.tem_image_features.norm(dim=-1, keepdim=True)
                        self.tem_logits = self.logit_scaling * self.tem_image_features @ self.text_features.t()
                        self.tem_similarity = (100.0 * self.tem_image_features @ self.text_features.T.half()).softmax(dim=-1)
                        _, self.tem_indices = self.tem_similarity[0].topk(1)

                sea_img = self.preprocess(search if batch_size == 1 else transforms.ToPILImage()(search[i])).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    sea_feat = self.clip_model.encode_image(sea_img)
                    sea_feat = self.ratio * self.adapter(sea_feat) + (1 - self.ratio) * sea_feat
                    sea_feat /= sea_feat.norm(dim=-1, keepdim=True)
                    sea_logits = self.logit_scaling * sea_feat @ self.text_features.t()
                    self.sem_ttfum.append(sea_logits)
                    mean_sea_logits = torch.stack(list(self.sem_ttfum)).mean(dim=0)
                    attention = self.softmax(torch.Tensor([-torch.norm(self.tem_logits - mean_sea_logits)])).unsqueeze(0)
                    tem_cls = self.text[self.tem_indices]
                    sea_cls = self.text[sea_logits.topk(1)[1]]

            class_des = torch.cat([class_des, tem_cls], dim=0)
            search_des = torch.cat([search_des, sea_cls], dim=0)
            attention_des = torch.cat([attention_des, attention.cuda()], dim=0)

        return class_des.view(batch_size, 1, 768, 1, 1), search_des.view(batch_size, 1, 768, 1, 1), attention_des
