from datetime import datetime

from tensorboardX import SummaryWriter
import utils
import torch
from engine import train_one_epoch, evaluate
from datasetload import ThyroidNoduleDataset, get_transform
from model import get_instance_segmentation_model
import os


def main():
    def save_pth(model, optimizer, epochs, ckpt_path, **kwargs):
        checkpoint = {}
        checkpoint["model"] = model.state_dict()
        checkpoint["optimizer"] = optimizer.state_dict()
        checkpoint["epochs"] = epochs
        checkpoint['lr_scheduler'] = lr_scheduler.state_dict()

        for k, v in kwargs.items():
            checkpoint[k] = v

        prefix, ext = os.path.splitext(ckpt_path)
        ckpt_path = "{}-{}{}".format(prefix, epochs, ext)
        torch.save(checkpoint, ckpt_path)

    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(0))
    log_dir = os.path.join(save_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(log_dir=log_dir)
    pth_path = "./save_weights/Maskrcnn-model.pth"  # 模型保存路径

    dataset = ThyroidNoduleDataset('train_data', get_transform(train=True))
    dataset_test = ThyroidNoduleDataset(
        'train_data', get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-100])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # the dataset has two classes only - background and person
    num_classes = 2

    # get the model using the helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # the learning rate scheduler decreases the learning rate by 10x every 3
    # epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.1)

    # training
    num_epochs = 15
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        _, loss = train_one_epoch(
            model, optimizer, data_loader, device, epoch, print_freq=10)
        writer.add_scalar('train-Loss', loss, global_step=epoch)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        # if epoch in range(num_epochs)[0:10]:
        save_pth(model, optimizer, epoch, pth_path)


if __name__ == '__main__':
    main()
