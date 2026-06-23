import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import numpy as np
import cv2,math,sys
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchvision.ops import DeformConv2d

class Logger(object):
    def __init__(self, filename="train_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# --- 数据集和模型配置 ---
DATA_DIR = 'data/512/'
ENCODER_NAME = 'se_resnext50_32x4d'
PRETRAINED_PATH = 'weights/DeepLaV3+_DECA.pth'
# PRETRAINED_PATH = 'weights/se_resnext50_32x4d-a260b3a4.pth'
# PRETRAINED_PATH = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# 从文件中读取类别
try:
    with open('data/labels_simple.txt', 'r') as f:
        data = f.readlines()
    CLASSES = [cl.strip() for cl in data]
except FileNotFoundError:
    print("Warning: 'data/labels_simple.txt' not found. Using default classes.")
    CLASSES = ['background', 'class1', 'class2']





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
    
class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, encoder_channels, out_channels=256):
        super().__init__()

        self.aspp = ASPP(
            in_channels=encoder_channels[5],
            out_channels=out_channels
        )

        # stage1低层特征
        low_level_channels = encoder_channels[2]

        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(
                out_channels + 48,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):

        # [input, stem, stage1, stage2, stage3, stage4]

        f_stage1 = features[2]
        f_stage4 = features[5]

        x = self.aspp(f_stage4)

        # 上采样到stage1尺寸
        x = F.interpolate(
            x,
            size=f_stage1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        low = self.low_level_conv(f_stage1)

        x = torch.cat([x, low], dim=1)

        x = self.fuse(x)

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

        if os.path.exists(encoder_weights_path):
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

        # self.decoder = DeepLabV3PlusDecoder(encoder_channels=encoder_channels, out_channels=256)

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




class Dataset(BaseDataset):
    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.class_values = [CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float32')
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return image, mask

    def __len__(self):
        return len(self.ids)



# ---------------------------------------------------------------
### 图像增强

def get_training_augmentation():
	train_transform = [

		albu.HorizontalFlip(p=0.5),

		albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

		albu.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
		albu.RandomCrop(height=512, width=512),

		albu.GaussNoise(p=0.2),
		albu.Perspective(p=0.5),

		albu.OneOf(
			[
				albu.CLAHE(p=1),
				albu.RandomBrightnessContrast(p=1),
				albu.RandomGamma(p=1),
			],
			p=0.9,
		),

		albu.OneOf(
			[
				albu.Sharpen(p=1),
				albu.Blur(blur_limit=3, p=1),
				albu.MotionBlur(blur_limit=3, p=1),
			],
			p=0.9,
		),

		albu.OneOf(
			[
				albu.RandomBrightnessContrast(p=1),
				albu.HueSaturationValue(p=1),
			],
			p=0.9,
		),
	]
	return albu.Compose(train_transform)


def get_validation_augmentation():
	"""调整图像使得图片的分辨率长宽能被32整除"""
	test_transform = [
		albu.PadIfNeeded(512, 512)
	]
	return albu.Compose(test_transform)

def get_preprocessing(preprocessing_fn=None):
    _transform = []
    if preprocessing_fn:
        _transform.append(albu.Lambda(image=preprocessing_fn))
    _transform.append(ToTensorV2())
    return albu.Compose(_transform)

if __name__ == '__main__':
    sys.stdout = Logger("train_log.txt")
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs. Using device: {DEVICE}")
    print(f"Number of classes: {len(CLASSES)}")

   # --- 1. 模型创建和权重加载 ---
    model = DeepLabV3Plus_DEConv(
        num_classes=len(CLASSES),
        encoder_name=ENCODER_NAME,
        encoder_weights_path=PRETRAINED_PATH
    )
   
    
    if os.path.exists(PRETRAINED_PATH):
        print(f"Loading pretrained encoder weights from: {PRETRAINED_PATH}")
        try:
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
            print("更新模型参数量：", len(updated_state_dict))
            print("Encoder weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}. Training with random encoder weights.")
    else:
        print("No local pretrained weights found or specified. Training with random encoder weights.")

    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER_NAME, 'imagenet')
    
    # --- 2. 数据集和数据加载器 ---
    x_train_dir = os.path.join(DATA_DIR, 'train', 'images')
    y_train_dir = os.path.join(DATA_DIR, 'train', 'masks')
    x_valid_dir = os.path.join(DATA_DIR, 'val', 'images')
    y_valid_dir = os.path.join(DATA_DIR, 'val', 'masks')

    train_dataset = Dataset(
        x_train_dir, y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )
    valid_dataset = Dataset(
        x_valid_dir, y_valid_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    batch_size = 2 * num_gpus if num_gpus > 0 else 2
    # batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Training images:   {len(train_dataset)}")
    print(f"Validation images: {len(valid_dataset)}")
    print(f"Batch size per step: {batch_size}")

    ### 3. 损失函数###
    loss = smp.losses.DiceLoss(mode='multilabel')
    metrics = [smp.metrics.iou_score]
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)

    ### 4. 循环训练########
    max_score = 0
    model.to(DEVICE)
    for i in range(120):
        print(f'\nEpoch: {i}')
        model.train()
        train_epoch_loss = 0
        train_epoch_iou = 0
        with tqdm(train_loader, desc='Train', unit='batch') as tepoch:
            for image, mask in tepoch:
                tepoch.set_description(f"Epoch {i} [Train]")
                image = image.to(DEVICE).float()
                mask = mask.to(DEVICE).permute(0, 3, 1, 2)

                optimizer.zero_grad()
                output = model(image)
                loss_train = loss(output, mask)
                loss_train.backward()
                optimizer.step()

                tp, fp, fn, tn = smp.metrics.get_stats(output, mask.long(), mode='multilabel', threshold=0.5)
                iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')

                train_epoch_loss += loss_train.item()
                train_epoch_iou += iou_score.item()

                tepoch.set_postfix(loss=loss_train.item(), iou=iou_score.item())

        avg_train_loss = train_epoch_loss / len(train_loader)
        avg_train_iou = train_epoch_iou / len(train_loader)
        print(f"Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}")

        model.eval()
        valid_epoch_loss = 0
        valid_epoch_iou = 0
        valid_epoch_precision = 0
        valid_epoch_recall = 0
        valid_epoch_f1_score = 0

        with torch.no_grad():
            with tqdm(valid_loader, desc='Valid', unit='batch') as vepoch:
                for image, mask in vepoch:
                    vepoch.set_description(f"Epoch {i} [Valid]")
                    image = image.to(DEVICE).float()
                    mask = mask.to(DEVICE).permute(0, 3, 1, 2)

                    output = model(image)
                    loss_val = loss(output, mask)

                    tp, fp, fn, tn = smp.metrics.get_stats(output, mask.long(), mode='multilabel', threshold=0.5)
                    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')
                    precision_score = smp.metrics.functional.precision(tp, fp, fn, tn,reduction='micro')
                    recall_score = smp.metrics.functional.recall(tp, fp, fn, tn, reduction='micro')
                    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro')

                    valid_epoch_loss += loss_val.item()
                    valid_epoch_iou += iou_score.item()
                    valid_epoch_precision += precision_score.item()
                    valid_epoch_recall += recall_score.item()
                    valid_epoch_f1_score += f1_score.item()

                    vepoch.set_postfix(loss=loss_val.item(), iou=iou_score.item(), precision = precision_score.item(), recall = recall_score.item(),f1_score = f1_score.item())

        avg_valid_loss = valid_epoch_loss / len(valid_loader)
        avg_valid_iou = valid_epoch_iou / len(valid_loader)
        avg_valid_precision = valid_epoch_precision/ len(valid_loader)
        avg_valid_recall = valid_epoch_recall / len(valid_loader)
        avg_valid_f1_score = valid_epoch_f1_score / len(valid_loader)

        print(f"Valid Loss: {avg_valid_loss:.4f}, Valid IoU: {avg_valid_iou:.4f}, Valid Precision: {avg_valid_precision:.4f}, Valid Recall: {avg_valid_recall:.4f}, Valid f1score: {avg_valid_f1_score:.4f}")
        lr_scheduler.step()
  

        if max_score < avg_valid_iou:
            max_score = avg_valid_iou
            if not os.path.exists('./weights'):
                os.makedirs('./weights')
            if isinstance(model, nn.DataParallel):
                state_dict_to_save = model.module.state_dict()
            else:
                state_dict_to_save = model.state_dict()
            torch.save(state_dict_to_save, f'./weights/best_model_epoch_{i}.pth')
            torch.save(state_dict_to_save, './best_model.pth')
            print('Model saved!')
