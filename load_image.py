import torch
import cv2
import numpy as np
from scipy import misc
from torch.autograd import Variable
import sys
import os

if len(sys.argv) < 4:
    print("please input : python load_image.py [file path] [result path] [dataset kind(A or B)]")
    exit()

file_path = sys.argv[1]
result_path = sys.argv[2]
dataset_kind = sys.argv[3]


model_path = os.path.dirname(os.path.abspath(__file__))  + '/models/'   # 이 파일이 존재한 곳의 models 폴더 경로
file_name = os.path.splitext(os.path.basename(file_path))[0] # 파일명에서 확장자를 분리함

epoch = "-7.0"
#epoch = "-24.0"

print(file_path)
print(result_path)
print(dataset_kind)
print(file_name)

def get_model():
    #torch.nn.Module.dump_patches = True
    generator_A = torch.load(os.path.join(model_path, 'model_gen_A') + epoch)
    generator_B = torch.load(os.path.join(model_path, 'model_gen_B') + epoch)

    if torch.cuda:
        generator_A = generator_A.cuda()
        generator_B = generator_B.cuda()

    return generator_A, generator_B

# path 불러오기
def get_real_image(image_size=64):
    images = []

    print(file_path)
    image = cv2.imread(file_path)  # fn이 한글 경로가 포함되어 있으면 제대로 읽지 못함. binary로 바꿔서 처리하는 방법있음

    if image is None:
        print("None")

     # image를 image_size(default=64)로 변환
    image = cv2.resize(image, (image_size, image_size))
    image = image.astype(np.float32) / 255.
    image = image.transpose(2, 0, 1)
    images.append(image)

    if images:
        print("push the stack")
        images = np.stack(images)
    else:
        print("error, images is emtpy")

    return images


def save_image(kind, image):

    print("save image")
    image = image[0].cpu().data.numpy().transpose(1, 2, 0) * 255.
    misc.imsave(os.path.join(result_path,  file_name + "_" + kind + '.jpg'), image.astype(np.uint8)[:, :, ::-1])



print("start")

generator_A, generator_B = get_model()
image = get_real_image()

A = Variable(torch.FloatTensor(image))

if torch.cuda:
    A = A.cuda()

if dataset_kind == 'A':
    AB = generator_B(A)
else:
    AB = generator_A(A)
#ABA = generator_A(AB)

save_image("A", A)
save_image("AB", AB)
#save_image("ABA", ABA)

print("finish")
