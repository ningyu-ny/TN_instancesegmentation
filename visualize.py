import torch
import matplotlib.pyplot as plt
from matplotlib import patches
def xyxy2xywh(box):
    new_box = torch.zeros_like(box)
    new_box[:, 0] = box[:, 0]
    new_box[:, 1] = box[:, 1]
    new_box[:, 2] = box[:, 2] - box[:, 0]
    new_box[:, 3] = box[:, 3] - box[:, 1]
    return new_box


def factor(n, base=1):
    base = base * 0.7 ** (n // 6) # mask 0.8
    i = n % 6
    if i < 3:
        f = [0, 0, 0]
        f[i] = base
    else:
        base /= 2
        f = [base, base, base]
        f[i - 3] = 0
    return f


def show(images, targets=None,position=None):
    if isinstance(images, (list, tuple)):
        for i in range(len(images)):
            show_single(images[i], targets[i] if targets else targets)
    else:
        show_single(images, targets,position)


def show_single(image, target, position):
    """
    Show the image, with or without the target
    Arguments:
        image (Tensor[3, H, W])
        target (Dict[Tensor])
    """
    image = image.clone()
    # result=target[0]
    result = target
    if result and "masks" in result:
        # masks = result["masks"].unsqueeze(1)
        if  'scores' in result:
            idx = result['scores'] > 0.75
            num = idx.count_nonzero().item()
        else:
            num=result['masks'].shape[0]

        masks = result["masks"][:num]
        # masks = masks.repeat(1, 3, 1, 1)
        for i, m in enumerate(masks):
            f = torch.tensor(factor(i, 0.8)).reshape(3, 1, 1).to(image)
            value = f * m
            image += value

    ax = plt.subplot(2,1,position)
    image = image.clamp(0, 1)
    im = image.cpu().numpy()
    im = im[0, :]
    ax.imshow(im.transpose(1, 2, 0))  # RGB
    # H, W = image.shape[-2:]
    # ax.set_title("H: {}   W: {}".format(H, W))
    ax.axis("off")

    if target:
        if "labels" in target:
            # if classes is None:
            #     raise ValueError("'classes' should not be None when 'target' has 'labels'!")
            tags = {l: i for i, l in enumerate(tuple(set(target["labels"].tolist())))}

        index = 0
        if "boxes" in result:
            boxes = result["boxes"][:num]
            boxes = xyxy2xywh(boxes).cpu().detach()
            for i, b in enumerate(boxes):
                if "labels" in result:
                    l = result["labels"][i].item()
                    # index = tags[l]
                    # txt = classes[l]
                    # if "scores" in result:
                    #     s = result["scores"][i]
                    #     s = round(s.item() * 100)
                    #     txt = "{} {}%".format(txt, s)
                    # ax.text(
                    #     b[0], b[1], txt, fontsize=9, color=(1, 1, 1),
                    #     horizontalalignment="left", verticalalignment="bottom",
                    #     bbox=dict(boxstyle="square", fc="black", lw=1, alpha=1)
                    # )

                rect = patches.Rectangle(b[:2], b[2], b[3], linewidth=2, edgecolor=factor(index), facecolor="none")
                ax.add_patch(rect)

    plt.show()
