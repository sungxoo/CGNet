import torch
import torchvision
import torchvision.transforms as transforms
from option import Option

def trainval(opt):
    device = torch.device(opt.device)

    train_transform = transforms.Compose([transforms.RandomResizedCrop(scale=opt.scale_limit),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.Normalize(mean=Dataset.mean, std=Dataset.std),
                                          transforms.ToTensor()])

    val_transform = transforms.Compose([transforms.Normalize(mean=Dataset.mean, std=Dataset.std),
                                        transforms.ToTensor()])

if __name__ == '__main__':
    trainval(Option().parse())
