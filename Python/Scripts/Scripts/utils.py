import torch
import numpy as np
import os
import base64
from io import BytesIO
import cv2
import sys
import importlib.util as util
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn.metrics as skmetrics


class TransformToUint(object):

    def __init__(self):
        self.v_max = 0
        self.v_min = 0

    def __call__(self, data, fitting):
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if fitting:
            self.v_max = data.max()
            self.v_min = data.min()
            #self.v_max = np.percentile(out, 98)
            #self.v_min = np.percentile(out, 2)

        
        
        data = (data - self.v_min) / (self.v_max - self.v_min + 1e-6)
        # boost dark regions
        stretch_constant = 10
        #data = data * stretch_constant
        #data = np.log(data + 1) / np.log(stretch_constant + 1)
        
        data = (data*255.).clip(0, 255)
        data = data.astype("uint8")
        return data
        

def tensor_to_string(x):
    # tensor has shape (C, H, W) with channels RGB
    if x.shape[0] == 3:
        x = x[[2, 1, 0]] # sort color channels to BGR
    if not isinstance(x, np.ndarray):
        x = x.detach().cpu().numpy()
        x = x * 255
        x = x.astype("uint8")
    # if image.shape[0] == 1:
    #     image.expand(-1, 3, -1, -1)
    #if image.shape[0] == 3:
    #    image = image[np.array([2,1,0])] # for sorting color channels to a unity friendly format BGR
    x = x.transpose([1,2,0]) # put channel dimension to last, new shape (H, W, C)
    image_enc = encode_image(x)
    return image_enc


def encode_image(data):
    _, buffer = cv2.imencode('.png', data)
    return base64.b64encode(buffer).decode('utf-8')


def decode_image(string_data, n_channels=3):
    decoded = base64.b64decode(string_data)
    decoded = np.frombuffer(decoded, dtype=np.uint8)
    mode = cv2.IMREAD_COLOR if n_channels==3 else cv2.IMREAD_GRAYSCALE
    image = cv2.imdecode(decoded, mode)
    if n_channels==3:
        image = image[:, :, [2, 1, 0]] # sort color channels
    return image


def get_module(module_path):
    module_name = os.path.basename(module_path)[:-3]
    
    #if module_path[0]!="/":
    #    module_path = '/'.join([os.getcwd(), module_path])
    
    spec = util.spec_from_file_location(module_name, module_path)
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def normalize(activations):
    # Normalize such that the values are between 0 and 1
    minimum = activations.min()
    maximum = activations.max()
    activations = (activations - minimum) / (maximum - minimum + 1e-8)
    return activations, minimum, maximum


def draw_confusion_matrix(targets, preds, labels=[], label_of_positive=None, vmax=None, show_numbers=True, show_colorbar=True, figsize=(7, 7)):
    conf_mat = skmetrics.confusion_matrix(targets, preds)
    n_classes = len(conf_mat[0])
    
    if labels:
        assert(n_classes == len(labels)), "Labels do not fit number of classes! n_classes: " + str(n_classes) + " labels: " + ', '.join(labels)
    
    if label_of_positive is not None and label_of_positive > 0:
        #permute conf_mat such that the 'positive' class is at top left corner
        assert(labels)
        conf_mat = bring_to_topleft_corner(conf_mat, label_of_positive)
        labels[label_of_positive], labels[0] = labels[0], labels[label_of_positive] 
    
    colors = ["bisque", "darkorange"]
    fig, ax = plt.subplots(figsize=figsize)
    if vmax is None:
        vmax = conf_mat.max()
    myimage = ax.imshow(conf_mat, cmap=mpl.colors.LinearSegmentedColormap.from_list("mycmap", colors), norm=mpl.colors.Normalize(vmin=0, vmax=vmax, clip=True), origin='upper')
    if show_colorbar:
        fig.colorbar(myimage, ticks=np.linspace(0, vmax, 6))
    if show_numbers:
        for i in range(n_classes):
            for j in range(n_classes):
                plt.text(x=j, y=i, s=f"{conf_mat[i][j]:{1}}", ha='center', va='center', fontsize=11)
    ax.set_xticks(np.arange(n_classes))
    if labels:
        ax.set_xticklabels(labels, ha='center', va='center')
        ax.set_yticklabels(labels, rotation=90, ha='center', va='center')
    ax.set_yticks(np.arange(n_classes))
    ax.set_ylim(bottom=n_classes - 0.5, top=-0.5)
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    labeltop=True) # labels along the bottom edge are off
    plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,
    labelleft=True) # labels along the bottom edge are off
    ax.set(xlabel='predicted', ylabel='actual')
    fig.tight_layout()
    return fig, ax


def bring_to_topleft_corner(conf_mat, label_of_positive):
    if label_of_positive == 0:
        return conf_mat
    conf_mat = conf_mat.copy()
    conf_mat[np.array([0, label_of_positive])] = conf_mat[np.array([label_of_positive, 0])]
    conf_mat[:,np.array([0, label_of_positive])] = conf_mat[:,np.array([label_of_positive, 0])]
    return conf_mat


def get_image_from_fig(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    image_encoded = base64.b64encode(image_png).decode('utf-8')
    return image_encoded
