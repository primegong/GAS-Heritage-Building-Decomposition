import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import matplotlib
matplotlib.use("Agg")
import numpy as np
import cv2, webcolors, colorsys, random,sys, math
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import cm
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset as BaseDataset
import imageio
from PIL import Image
import torch.nn as nn
from torchvision.ops import DeformConv2d
import torch.nn.functional as F
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# CLASSES = ['roof',
#             'wall',
#             'window',
#             'stone carving',
#             'door',
#             'glass',
#             'balcony',
#             'chimeny']



data = open('data/labels_simple.txt','r').readlines()
CLASSES = [cl.strip() for cl in data]




##—————————————————————————————————————————————————————————————— model ————————————————————————————————————————————————————————————————————

class DeformConv(nn.Module):
    """
    形变空洞卷积：带 dilation 的 DeformConv2d
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False):
        super(DeformConv, self).__init__()
        padding = dilation * (kernel_size // 2)

        # offset conv 生成偏移量
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=3,
            stride=stride,
            padding=1
        )

        # deformable conv
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        offset = self.offset_conv(x)
        out = self.deform_conv(x, offset)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ECAAttention(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super(ECAAttention, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class DenseECAFusion(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(DenseECAFusion, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels1 + in_channels2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.eca = ECAAttention(out_channels)

    def forward(self, x1, x2):
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv_block(x)
        x = self.eca(x)
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )
        for rate in rates:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels + out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]

        pooled = self.image_pool(x)
        pooled = F.interpolate(pooled, size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(pooled)

        res = torch.cat(res, dim=1)
        return self.project(res)


class CustomDecoder(nn.Module):
    def __init__(self, encoder_channels, out_channels=256):
        super(CustomDecoder, self).__init__()
        # smp encoder_channels 顺序: [input, stage1, stage2, stage3, stage4]
        # 需要 stage1, stage2, stage3
        c_stage1, c_stage2, c_stage3 = encoder_channels[2], encoder_channels[3], encoder_channels[4]
        self.aspp = ASPP(in_channels=encoder_channels[5], out_channels=out_channels)
        self.fusion1 = DenseECAFusion(out_channels, c_stage3, out_channels)
        self.fusion2 = DenseECAFusion(out_channels, c_stage2, out_channels)
        self.fusion3 = DenseECAFusion(out_channels, c_stage1, out_channels)

    def forward(self, features):
        # smp features 顺序: [input, stem, stage1, stage2, stage3, stage4]
        f_stage1, f_stage2, f_stage3, f_stage4 = features[2], features[3], features[4], features[5]
        x = self.aspp(f_stage4)
        x = self.fusion1(x, f_stage3)
        x = self.fusion2(x, f_stage2)
        x = self.fusion3(x, f_stage1)
        return x

class DeepLabV3Plus_DEConv(nn.Module):
    def __init__(self, num_classes, encoder_name='se_resnext50_32x4d', encoder_weights_path=None):
        super(DeepLabV3Plus_DEConv, self).__init__()

        self.encoder = smp.encoders.get_encoder(
            name=encoder_name,
            in_channels=3,
            depth=5,
            weights=None,
        )
        self._modify_encoder()

        if encoder_weights_path and os.path.exists(encoder_weights_path):
            print(f"Loading local pretrained weights from: {encoder_weights_path}")
            try:
                state_dict = torch.load(encoder_weights_path, map_location=DEVICE)
                # 使用 strict=False 以防权重文件包含分类头等不需要的键
                self.encoder.load_state_dict(state_dict, strict=False)
                self.encoder.to(DEVICE)
                print("Local encoder weights loaded successfully.")
            except Exception as e:
                print(f"Error loading local weights: {e}. Encoder is randomly initialized.")
        else:
            print("Warning: No local pretrained path provided or path not found. Encoder is randomly initialized.")
        
        encoder_channels = self.encoder.out_channels
        self.decoder = CustomDecoder(encoder_channels=encoder_channels, out_channels=256)
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    def _modify_encoder(self):
        # 遍历 stage4 (layer4)，把其中的 3x3 conv 换成 DeformDilatedConv
        from pretrainedmodels.models.senet import SEResNeXtBottleneck
        for i, layer_name in enumerate(['layer1','layer2','layer3', 'layer4']):
            layer = getattr(self.encoder, layer_name)
            for block in layer:
                if isinstance(block, SEResNeXtBottleneck):
                    original_conv = block.conv2
                    in_channels = original_conv.in_channels
                    out_channels = original_conv.out_channels
                    stride = original_conv.stride[0]
                    dilation = original_conv.dilation[0]
                    block.conv2 = DeformConv(in_channels, out_channels, kernel_size=3, stride=stride, dilation=dilation)
            

    def forward(self, x):
        input_size = x.shape[2:]
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        logits = self.segmentation_head(decoder_output)
        logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        return logits





##—————————————————————————————————————————————— model  end —————————————————————————————————————————————————————————————————————————————————


def get_colour_name(requested_colour):
	try:
		closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
	except ValueError:
		closest_name = closest_colour(requested_colour)
		actual_name = None
	return actual_name, closest_name

def randomColor(labels):
	num = len(labels)
	colors = [colorsys.hsv_to_rgb(i/num, 1, 1) for i in range(num)]
	colors = [(int(c[0]) *255, int(c[1]) *255, int(c[2]) *255) for c in colors]
	return colors


def random_colors(N, bright=True):
	"""
	Generate random colors.
	To get visually distinct colors, generate them in HSV space then
	convert to RGB.
	"""
	brightness = 1.0 if bright else 0.7
	hsv = [(i / N, 1, brightness) for i in range(N)]
	colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
	random.shuffle(colors)
	return colors


def apply_mask(image, mask, color, alpha=0.5):
	"""Apply the given mask to the image.
	"""
	for c in range(3):
		image[:, :, c] = np.where(mask == 1,
								  image[:, :, c] *
								  (1 - alpha) + alpha * color[c] * 255,
								  image[:, :, c])
	return image


def display_instances(image, pr_mask,savefig):

	fig, ax = plt.subplots(1, figsize=(16, 16))

	N = len(CLASSES)
	# Generate random colors
	colors = random_colors(N)
	masked_image = image.astype(np.uint32).copy()
	for i in range(1,N):
		color = colors[i]

		# Mask
		mask = np.zeros((512,512))
		mask = np.where(pr_mask==i,1, mask)
		masked_image = apply_mask(masked_image, mask, color)

		# Mask Polygon
		# Pad to ensure proper polygons for masks that touch image edges.
		padded_mask = np.zeros(
			(mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
		padded_mask[1:-1, 1:-1] = mask
		contours = find_contours(padded_mask, 0.5)
		for verts in contours:
			# Subtract the padding and flip (y, x) to (x, y)
			verts = np.fliplr(verts) - 1
			p = Polygon(verts, facecolor="none", edgecolor=color)
			ax.add_patch(p)
			cx = np.mean(verts[:,0])
			cy = np.mean(verts[:,1])
			ax.text(cx, cy, CLASSES[i],
				color='w', size=15, backgroundcolor="none")
	ax.imshow(masked_image.astype(np.uint8))
	# 去边
	plt.subplots_adjust(top=1, bottom=0,left=0,right=1,hspace=0,wspace=0)
	plt.margins(0,0)
	plt.savefig(savefig)
	plt.close(fig)

# ---------------------------------------------------------------
### Dataloader

class Dataset(BaseDataset):
	"""CamVid数据集。进行图像读取，图像增强增强和图像预处理.

	Args:
		images_dir (str): 图像文件夹所在路径
		masks_dir (str): 图像分割的标签图像所在路径
		class_values (list): 用于图像分割的所有类别数
		augmentation (albumentations.Compose): 数据传输管道
		preprocessing (albumentations.Compose): 数据预处理
	"""

	def __init__(
			self,
			images_dir,
			# masks_dir,
			classes=None,
			augmentation=None,
			preprocessing=None,
	):
		self.ids = os.listdir(images_dir)
		self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]

		# convert str names to class values on masks
		self.class_values = [CLASSES.index(cls.lower()) for cls in classes]

		self.augmentation = augmentation
		self.preprocessing = preprocessing

	def __getitem__(self, i):

		# read data
		image = cv2.imread(self.images_fps[i])
		image = cv2.resize(image, (512, 512))   # 改变图片分辨率
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# 图像增强应用
		if self.augmentation:
			sample = self.augmentation(image=image)
			image = sample['image']

		# 图像预处理应用
		if self.preprocessing:
			sample = self.preprocessing(image=image)
			image = sample['image']

		return image

	def __len__(self):
		return len(self.ids)

# ---------------------------------------------------------------

def get_validation_augmentation():
	"""调整图像使得图片的分辨率长宽能被32整除"""
	test_transform = [
		albu.PadIfNeeded(512, 512),
		albu.Resize(height=512, width=512) #缩放至512大小
	]
	return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
	return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
	"""进行图像预处理操作

	Args:
		preprocessing_fn (callbale): 数据规范化的函数
			(针对每种预训练的神经网络)
	Return:
		transform: albumentations.Compose
	"""

	_transform = [
		albu.Lambda(image=preprocessing_fn),
		albu.Lambda(image=to_tensor),
	]
	return albu.Compose(_transform)


# 图像分割结果的可视化展示
def visualize(**images):
	"""PLot images in one row."""
	n = len(images)
	fig = plt.figure(figsize=(16, 5))
	for i, (name, image) in enumerate(images.items()):
		ax = fig.add_subplot(1, n, i + 1)
		ax.set_title(' '.join(name.split('_')).title())
		ax.imshow(image)
    # plt.imshow()
	return fig



# ---------------------------------------------------------------
if __name__ == '__main__':

	# DATA_DIR = './build/512/'
	# x_test_dir = os.path.join(DATA_DIR, 'val','images')

	x_test_dir = 'data/512/val/images'

	height = 512
	weight = 512

	ENCODER = 'se_resnext50_32x4d'
	ENCODER_WEIGHTS = 'imagenet'
	PRETRAINED_PATH = 'weights/DeepLaV3+_DECA.pth'

	ACTIVATION = 'softmax' # could be None for logits or 'softmax2d' for multiclass segmentation
	DEVICE = 'cuda'

	# 按照权重预训练的相同方法准备数据
	preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

	# # 加载smp已有的模型:DeepLabV3Plus, Unet, PSPNet
	# model = smp.PSPNet(
    #     encoder_name=ENCODER,
    #     encoder_weights=ENCODER_WEIGHTS,
    #     classes=len(CLASSES),
    #     activation=ACTIVATION,
    #     )
	

    # 加载自己的模型:DeepLabV3Plus_DEConv
	model = DeepLabV3Plus_DEConv(
        num_classes=len(CLASSES),
        encoder_name=ENCODER,
        encoder_weights_path=PRETRAINED_PATH
    )

    #### 加载全部模型权重 ##############
	state_dict = torch.load(PRETRAINED_PATH, map_location=DEVICE)       
    ## 去除 'encoder.'和'decoder.' 前缀，避免key不匹配，模型加载失败
	new_state_dict = {}
	for key in state_dict.keys():
		new_key = key
		if key.startswith('encoder.'):
			new_key = key[len('encoder.'):]  # 去掉 'encoder.' 前缀
		if key.startswith('decoder.'):
			new_key = key[len('decoder.'):]  # 去掉 'decoder.' 前缀
		new_state_dict[new_key] = state_dict[key]

	

	# 获取当前模型的 state_dict,检查是否逐层对应，并赋予权重
	current_state_dict = model.state_dict()
	print("模型总参数量：", len(current_state_dict))
	print("预训练模型总参数量：", len(new_state_dict.keys()))      

	# 创建一个字典用于存储需要更新的权重
	updated_state_dict = {}
	unexpected= {}

	# 遍历预训练权重，逐层加载
	for name, param in state_dict.items():
		# 检查当前模型是否有该层
		if name in current_state_dict:
			# 如果层匹配，则将预训练权重加载到当前模型的对应层
			updated_state_dict[name] = param
		else:
			print(f"Skipping layer {name} as it is not present in the current model.")
			unexpected[name] = param
			

	# 将更新后的权重加载到模型中
	model.load_state_dict(updated_state_dict, strict=False)

	model = model.to(DEVICE)
	model.eval()
    #### 加载全部模型权重完毕 ##############




	# 创建检测数据集
	predict_dataset = Dataset(
		x_test_dir,
		augmentation=get_validation_augmentation(),
		preprocessing=get_preprocessing(preprocessing_fn),
		classes=CLASSES,
	)

	image_names = predict_dataset.images_fps

	# 对检测图像进行图像分割并进行图像可视化展示
	predict_dataset_vis = Dataset(
		x_test_dir,
		augmentation=get_validation_augmentation(),
		classes=CLASSES,
	)


	save_dir = './val_results/DeepLabV3Plus_DEConv/'
	vis_dir = save_dir + '/result_vis'
	if not os.path.exists(vis_dir):
		os.makedirs(vis_dir)

	result_dir = save_dir + '/result_multiclass'
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)

	vis_combine_dir = save_dir + '/result_combine_vis'
	if not os.path.exists(vis_combine_dir):
		os.makedirs(vis_combine_dir) 
	
	

	##### 固定颜色
	cmap = plt.get_cmap('viridis')
	cmaplist = [cmap(i) for i in range(cmap.N)]
	N = len(cmaplist)
	interval  = int(np.floor(N/10))
	colors = cmaplist[0:N:interval]
	colors.append(cmaplist[-1])

	with torch.no_grad():
		for i in range(len(predict_dataset)):
			# 原始图像image_vis
			image_vis = predict_dataset_vis[i].astype('uint8')
			image = predict_dataset[i]
			name = image_names[i].replace('\\','/').split('/')[-1]
			print(name)

			# 通过图像分割得到的0-1图像pr_maskpredict
			x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
			# pr_mask = model.predict(x_tensor)
			pr_mask = model.forward(x_tensor)
			pr_mask = (pr_mask.squeeze().cpu().numpy().round())
			pr_mask = np.argmax(pr_mask, axis=0)


			# 恢复图片原来的分辨率
			image_vis = cv2.resize(image_vis, (weight, height))
			pr_mask = cv2.resize(pr_mask, (weight, height))
			# 保存图像分割后的黑白结果图像
			# imageio.imwrite('result/' + str(i)+'.png', pr_mask)
			# 原始图像和图像分割结果的可视化展示





			### 保存黑白结果
			mask_image = Image.fromarray(pr_mask.astype('uint8'))
			mask_image.save(os.path.join(result_dir, name))

			#### 叠加显示
			savefig = vis_combine_dir + '/' + name
			display_instances(image_vis, pr_mask, savefig)


			## 保存彩色结果
			group_ids = np.unique(pr_mask.flatten())
			colormap = np.zeros((512,512,4))
			for gi in group_ids:
				[y,x]=np.where(pr_mask==gi)
				colormap[y,x,:] = list(colors[gi])

			#### 原始显示方式
			fig2 = visualize(
				image=image_vis,
				predicted_mask=colormap
			)
            
			fig2.savefig(vis_dir + '/' + name)
			plt.close(fig2)
			plt.close('all')
			# utils.lblsave(os.path.join(result_dir, image_name), pr_mask)

