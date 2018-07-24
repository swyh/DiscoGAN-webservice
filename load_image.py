import torch
import cv2
import numpy as np
from scipy import misc
from torch.autograd import Variable
import sys

if len(sys.argv) < 4:
    print("please input file path, result path, dataset kind")
    exit()

file_path = sys.argv[1]
result_path = sys.argv[2]
dataset_kind = sys.argv[3]

#result_path = 'C:/DiscoGAN/DiscoGAN2/discogan/test/result/'
#model_path ='C:/DiscoGAN/DIscoGAN2/discogan/models/handbags2shoes/discogan/'
#model_path ='C:/DiscoGAN/DIscoGAN2/discogan/models/face/'
model_path ='./models/'

#epoch = "-7.0"
epoch = "-24.0"



def get_model():
    generator_A = torch.load(model_path +'model_gen_A' + epoch)
    generator_B = torch.load(model_path +'model_gen_B' + epoch)
    discriminator_A = torch.load(model_path +'model_dis_A' + epoch)
    discriminator_B = torch.load(model_path +'model_dis_B' + epoch)

    if torch.cuda:
        generator_A = generator_A.cuda()
        generator_B = generator_B.cuda()
        discriminator_A = discriminator_A.cuda()
        discriminator_B = discriminator_B.cuda()

    return generator_A, generator_B, discriminator_A, discriminator_B

# path 불러오기
def get_real_image(image_size=64):

    print(file_path)
    image = cv2.imread(file_path)  # fn이 한글 경로가 포함되어 있으면 제대로 읽지 못함. binary로 바꿔서 처리하는 방법있음

    if image is None:
        print("None")

     # image를 image_size(default=64)로 변환
    image = cv2.resize(image, (image_size, image_size))
    image = image.astype(np.float32) / 255.
    image = image.transpose(2, 0, 1)

    return image


def save_image(name, image):

    print("save image")
    image = image.cpu().data.numpy().transpose(1, 2, 0) * 255.
    misc.imsave(result_path + "_" + name + '.jpg', image.astype(np.uint8)[:, :, ::-1])



print("start")

generator_A, generator_B, discriminator_A, discriminator_B = get_model()
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
