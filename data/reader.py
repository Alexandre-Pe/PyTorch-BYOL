import os
import torch
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def Decode_MPII(line):
    """
    Dimension 1: image file path and name.
    Dimension 2~3: Gaze location on the screen coordinate in pixels, the actual screen size can be found in the "Calibration" folder.
    Dimension 4~15: (x,y) position for the six facial landmarks, which are four eye corners and two mouth corners.
    Dimension 16~21: The estimated 3D head pose in the camera coordinate system based on 6 points-based 3D face model, rotation and translation: we implement the same 6 points-based 3D face model in [1], which includes the four eye corners and two mouth corners.
    Dimension 22~24 (fc): Face center in the camera coordinate system, which is averaged 3D location of the 6 focal landmarks face model. Not it is slightly different with the head translation due to the different centers of head and face.
    Dimension 25~27 (gt): The 3D gaze target location in the camera coordinate system. The gaze direction can be calculated as gt - fc.
    Dimension 28: Which eye (left or right) is used for the evaluation subset.
    """
    anno = edict()
    anno.face = line[0]
    anno.gaze2d = [float(line[1]), float(line[2])]
    anno.head2d = [[float(line[3]), float(line[4])],
                   [float(line[5]), float(line[6])],
                   [float(line[7]), float(line[8])],
                   [float(line[9]), float(line[10])],
                   [float(line[11]), float(line[12])],
                   [float(line[13]), float(line[14])]
                  ]
    anno.head3d = [[float(line[15]), float(line[16]), float(line[17])],
                   [float(line[18]), float(line[19]), float(line[20])]]
    anno.facecenter = [float(line[21]), float(line[22]), float(line[23])]
    anno.gaze3d = [float(line[24]), float(line[25]), float(line[26])]
    anno.eye = line[27]
    return anno

def Decode_Diap(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d, anno.head3d = line[4], line[5]
    anno.gaze2d, anno.head2d = line[6], line[7]
    return anno

def Decode_Gaze360(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d = line[4]
    anno.gaze2d = line[5]
    return anno

def Decode_ETH(line):
    anno = edict()
    anno.face = line[0]
    anno.gaze2d = line[1]
    anno.head2d = line[2]
    anno.name = line[3]
    return anno

def Decode_RTGene(line):
    anno = edict()
    anno.face = line[0]
    anno.gaze2d = line[6]
    anno.head2d = line[7]
    anno.name = line[0]
    return anno

def Decode_Dict():
    mapping = edict()
    mapping.mpiigaze = Decode_MPII
    mapping.eyediap = Decode_Diap
    mapping.gaze360 = Decode_Gaze360
    mapping.ethtrain = Decode_ETH
    mapping.rtgene = Decode_RTGene
    return mapping

def long_substr(str1, str2):
    substr = ''
    for i in range(len(str1)):
        for j in range(len(str1)-i+1):
            if j > len(substr) and (str1[i:i+j] in str2):
                substr = str1[i:i+j]
    return len(substr)

def Get_Decode(name):
    mapping = Decode_Dict()
    keys = list(mapping.keys())
    name = name.lower()
    score = [long_substr(name, i) for i in keys]
    key  = keys[score.index(max(score))]
    return mapping[key]
    

class trainloader(Dataset): 
  def __init__(self, dataset, transform):

    # Read source data
    self.data = edict() 
    self.data.line = []
    self.data.root = dataset.image
    self.data.decode = Get_Decode(dataset.name)

    if isinstance(dataset.label, list):

      for i in dataset.label:

        with open(i) as f: line = f.readlines()

        if dataset.header: line.pop(0)

        self.data.line.extend(line)

    else:

      with open(dataset.label) as f: self.data.line = f.readlines()

      if dataset.header: self.data.line.pop(0)

    self.transforms = transform


  def __len__(self):

    return len(self.data.line)


  def __getitem__(self, idx):

    # Read souce information
    line = self.data.line[idx]
    line = line.strip().split(" ")
    anno = self.data.decode(line)

    img_path = os.path.join(self.data.root, anno.face)
    img = Image.open(img_path).convert('RGB')
    img = self.transforms(img)

    label = np.array(anno.gaze2d).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)

    return img, label

def loader(source, batch_size, transform, shuffle=True,  num_workers=0):
    dataset = trainloader(source, transform)
    print(f"-- [Read Data]: Source: {source.label}")
    print(f"-- [Read Data]: Total num: {len(dataset)}")
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load

