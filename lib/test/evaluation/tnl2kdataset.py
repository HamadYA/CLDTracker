import os

import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text, load_str

############
# current 00000492.png of test_015_Sord_video_Q01_done is damaged and replaced by a copy of 00000491.png
############


class TNL2kDataset(BaseDataset):
    """
    TNL2k test set
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.tnl2k_path
        self.sequence_list = self._get_sequence_list()
        # self.sequence_list = ['BianLian_video_03_done', 'Bullet_video_08_done', 'Cartoon_Robot_video_Z01_done', 'CrashCar_video_04', 'CartoonHuLuWa_video_04-Done']

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        # class_name = sequence_name.split('-')[0]
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        text_dsp_path = '{}/{}/language_original.txt'.format(self.base_path, sequence_name)
        text_dsp = load_str(text_dsp_path)

        frames_path = '{}/{}/imgs'.format(self.base_path, sequence_name)
        frames_list = [f for f in os.listdir(frames_path)]
        frames_list = sorted(frames_list)
        frames_list = ['{}/{}'.format(frames_path, frame_i) for frame_i in frames_list]

        # target_class = class_name
        return Sequence(sequence_name, frames_list, 'tnl2k', ground_truth_rect.reshape(-1, 4), object_class=text_dsp)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        # sequence_list = []
        # cover_list = []
        # for seq in os.listdir(self.base_path):
        #     if os.path.isdir(os.path.join(self.base_path, seq)):
        #         text_dsp_path = '{}/{}/language_original.txt'.format(self.base_path, seq)
        #         text_dsp = load_str(text_dsp_path)
        #         sequence_list.append(seq)
        
        sequence_list = [
                'test_019_Robot_x02_done',
                'test_029_Joker_video_X01_done',
                'test_030_JinglingWangzi_video_04_done',
                'test_032_Hellbaby_video_02_done',
                'test_037_Group_video_2_done',
                'test_044_catNight_video_B01_done',
                'thor_3-Done',
                'Thor_video_X04-Done',
                'Transform_video_Q18-Done',
                'Transform_video_X12-Done',
                'Transform_video_X14-Done',
                'Wonderwomen_video_11-Done',
                'Wonderwomen_video_13-Done',
                'advSamp_Chess_video_4-Done',
                'advSamp_Chess_video_6-Done',
                'advSamp_CrashCar_video_15-Done',
                'advSamp_DarkRider_video_08-Done',
                'advSamp_INF_bikemove1',
                'advSamp_INF_biketwo',
                'advSamp_INF_blackwoman',
                'advSamp_INF_boundaryandfast',
                'advSamp_monitor_bike5',
                'advSamp_monitor_bluegirl',
                'Assian_video_Z03_done',
                'BatMan_video_16_done',
                'BF5_Blade_video_01-Done',
                'BF5_Blade_video_03_done',
                'Cartoon_Angle_video_02_done',
                'Cartoon_BianSeLong_video_C01_done',
                'Cartoon_BirdHead_video_01_done',
                'Cartoon_Fox_video_01_done',
                'CartoonKobe_video_02_done',
                'CartoonKobe_video_03_done',
                'Cartoon_ManHead_video_03_done',
                'Cartoon_ManHead_video_Z01_done',
                'Cartoon_Mouse_video_04_done',
                'CartoonSanGUO_video_08',
                'CartoonSlamDUNK_video_01',
                'CartoonSlamDUNK_video_05_done',
                'CartoonSlamDUNK_video_11',
                'CartoonXiYouJi_video_01',
                'CartoonXiYouJi_video_03',
                'Chase_video_07',
                'Chase_video_U01-Done',
                'Chase_video_U02-Done',
                'Chase_video_X04-2',
                'CheerLeader_video_01-Done',
                'CheerLeader_video_05-Done',
                'Chess_video_16',
                'Chess_video_5-Done',
                'CM_mancrossandup_done',
                'CM_maninglass_done',
                'CM_manonboundary_done',
                'CM_manup_done',
                'CM_manypeople1_done',
                'CM_nightthreepeople_done',
                'CM_orangeman1_done',
                'CM_push_done',
                'CM_redcar_done',
                'CM_single1_done',
                'CM_straw_done',
                'CM_tallman_done',
                'CM_toy1_done',
                'CM_tree5_done',
                'CM_tricyclefaraway_done',
                'CM_twoperson_done',
                'CM_walking40_done',
                'CM_walkingman20_done',
                'CrashCar_video_15-Done',
                'Elephant_video_5',
                'Fight_game_011-Done',
                'Fight_game_022-Done',
                'Fight_game_027-Done',
                'Fire_video_02-Done',
                'Fire_video_08-Done',
                'FlyFish_video_06-Done',
                'FlyFish_video_08-Done',
                'Football_video_04',
                'Football_video_06',
                'Football_video_08',
                'Football_video_10',
                'gamGUN_11-Done',
                'gamGUN_33-Done',
                'gamGUN_35-Done',
                'Hulk_video_U01-Done',
                'IceBall_video_05-Done',
                'INF_biketwo',
                'INF_blackwoman',
                'INF_boundaryandfast',
                'INF_call',
                'INF_campus2',
                'INF_crowd4',
                'INF_diamond',
                'INF_dog1',
                'INF_dog11',
                'INF_elecbike2',
                'INF_flower1',
                'INF_glass2',
                'INF_guidepost',
                'INF_kite4',
                'INF_man22',
                'INF_manafterrain',
                'INF_manwithumbrella',
                'INF_manypeople',
                'INF_manypeople1',
                'INF_tree5',
                'INF_tricycle6',
                'INF_tricyclefaraway',
                'INF_twoperson',
                'INF_walkingman20',
                'INF_walkingnight',
                'INF_walkingwoman',
                'ironman_3-Done',
                'James_video_07',
                'James_video_09',
                'JapaMan_video_04-Done',
                'LeftWoman_video_C01',
                'Lightman_video_02',
                'maliao_11-Done',
                'ManHat_test_002_done',
                'manHead_59-Done',
                'Man_video_01-Done',
                'Man_video_03-Done',
                'Man_video_04L-Done',
                'MaoNv_video_11',
                'monitor_bike2',
                'monitor_bike5',
                'monitor_bluegirl',
                'monitor_car2',
                'monitor_couple',
                'monitor_couple2',
                'monitor_diamond',
                'monitor_dog',
                'monitor_E-bike4',
                'monitor_E-bike6',
                'monitor_manypeople2',
                'monitor_raincoatriding',
                'monitor_redman',
                'monitor_running',
                'Monkey_BBC_video_03-Done',
                'NBA_Bull_video_02-Done',
                'NBA_Bull_video_04-Done',
                'RacecarSingle_video_04-Done',
                'Roman_video18-Done',
                'Roman_video1-Done',
                'Soccer_video_08-Done',
                'SportGirl_video_05_done',
                'StarWar_TPS_video_12-Done',
                'StarWar_TPS_video_15L-Done',
                'Sunwufan_video_3-Done',
                'SuperMan_video_01_done',
                'SuperMan_video_02_done',
                'SuperMan_video_U02_done',
                'Teluoyi_video_1-Done',
                'Teluoyi_video_3-Done',
                'Terminator_video_01-Done',
                'Terminator_video_05-Done'
                ]
        return sequence_list


label = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                      'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                      'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                      'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                      'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                      'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                      'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                      'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                      'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                      'teddy bear', 'hair drier', 'toothbrush', 'nba', 'people', 'baseball', 'ball', 'tennis', 'phone', 'table']