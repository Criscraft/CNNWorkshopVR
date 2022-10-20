import asyncio
from tkinter import Image
from Scripts.ImageResource import ImageResource
import websockets
import json
import os
import sys
import time
import torch
import numpy as np
import Scripts.utils as utils
from Scripts.DLNetwork import FeatureVisualizationMode
from Scripts.utils import get_module

# Create global variables. They will be filled inplace.
network = None
dataset = None
noise_generator = None

def prepare_dl_objects(source):
    global dataset
    dataset = source.get_dataset()
    global network
    network = source.get_network()
    # Set the class names for the network. This is important if class names should be displayed for some channels.
    network.class_names = dataset.class_names
    global noise_generator
    noise_generator = source.get_noise_generator()

    data = torch.zeros(dataset.get_data_item(0, True)[0].shape)
    image_resource = ImageResource(id=0, mode=ImageResource.Mode.DATASET, data=data)
    network.initial_forward_pass(image_resource)


async def handler(websocket):
    async for message in websocket:
        response = ""
        event = json.loads(message)
        
        if event["resource"] == "request_dataset_images":
            response = request_dataset_images(event)
        elif event["resource"] == "request_forward_pass":
            response = request_forward_pass(event)
        elif event["resource"] == "request_architecture":
            response = request_architecture(event)
        elif event["resource"] == "request_image_data":
            response = request_image_data(event)
        
        if response:
            await websocket.send(response)


def request_dataset_images(event):
    image_resources = []
    n = event["n"]
    if n > 0:
        rand_inds = np.random.randint(len(dataset), size=n)
        for rand_ind in rand_inds:
            tensor, label = dataset.get_data_item(rand_ind, False)
            image_resources.append(ImageResource(
                id=int(rand_ind),
                data=utils.tensor_to_string(tensor),
                label=dataset.class_names[label],
                mode=ImageResource.Mode.DATASET,
            ).__dict__)
    response = {"resource" : "request_dataset_images", "image_resources" : image_resources}
    response = json.dumps(response, indent=1, ensure_ascii=True)
    print("send: " + "request_dataset_images")
    return response


def request_forward_pass(event):
    # Create image resource
    image_resource_dict = event["image_resource"]
    image_resource = ImageResource(
        id = image_resource_dict["id"],
        module_id = image_resource_dict["module_id"],
        channel_id = image_resource_dict["channel_id"],
        mode = ImageResource.Mode(image_resource_dict["mode"]),
    )

    # Create image for the image resource
    image = None
    if image_resource.mode == ImageResource.Mode.DATASET and image_resource.id >= 0:
        image, label = dataset.get_data_item(image_resource.id, False)
        image_resource.label = dataset.class_names[label]
    elif image_resource.mode == ImageResource.Mode.FEATURE_VISUALIZATION:
        image = network.try_load_feature_visualization(image_resource.module_id)
        if image is not None:
            image = image[image_resource.channel_id]
        else:
            raise RuntimeError
    elif image_resource.mode == ImageResource.Mode.NOISE:
        image = noise_generator.get_noise_image()
    elif image_resource.mode == ImageResource.Mode.ACTIVATION:
        raise RuntimeError
    else:
        image = torch.zeros(dataset.get_data_item(0, True)[0].shape)
    image_resource.data = image

    # Perform the forward pass
    network.forward_pass(image_resource)
    
    # Get network results and make response.
    posteriors, class_indices = network.get_classification_result()
    response = ""
    if posteriors is not None:
        response = {
            "class_names" : [dataset.class_names[ind] for ind in class_indices],
            "confidence_values" : [f"{item:.2f}" for item in posteriors],
        }
        response = {"resource" : "request_forward_pass", "results" : response}
        response = json.dumps(response, indent=1, ensure_ascii=False)
    print("send: " + "request_forward_pass")
    return response


def request_architecture(event):
    architecture_dict = network.get_architecture()
    response = {"resource" : "request_architecture", "architecture" : architecture_dict}
    response = json.dumps(response, indent=1, ensure_ascii=False)
    print("send: " + "request_architecture")
    return response


def request_image_data(event):
    image_resources = []
    module_resources = event["network_module_resource"]
    mode = event["mode"]
    if mode == "activation":
        module_id = module_resources["module_id"]
        activations = network.get_activation(module_id)[0]
        has_module_class_names = network.has_module_class_names(module_id)
        for channel_id, activation in enumerate(activations):
            image_resources.append(ImageResource(
                module_id=module_id,
                channel_id=channel_id,
                mode=ImageResource.Mode.ACTIVATION,
                label=dataset.class_names[channel_id] if has_module_class_names else "",
                data=utils.tensor_to_string(activation.unsqueeze(0)),
            ).__dict__)
    response = {"resource" : "request_image_data", "image_resources" : image_resources}
    response = json.dumps(response, indent=1, ensure_ascii=True)
    print("send: " + "request_image_data")
    return response


async def main():
    async with websockets.serve(handler, "", 8000, max_size=2**30, read_limit=2**30, write_limit=2**30):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    source_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "Projects", "HFNetMNIST", "get_dl_objects.py")
    get_dl_objects_module = get_module(source_path)
    prepare_dl_objects(get_dl_objects_module)
    asyncio.run(main())


"""
class NetworkResource:

    def on_get(self, req, resp):
        out = {
            "nnetworks" : len(networks),
            "ndatasets" : len(datasets),
            "nnoiseGenerators" : len(noise_generators),
        }
        resp.text = json.dumps(out, indent=1, ensure_ascii=False)
        print("send NetworkResource")


class NetworkArchitectureResource:

    def on_get(self, req, resp, networkid):
        network = networks[networkid]
        out = network.get_architecture()
        out['class_names'] = list(datasets[network.corresponding_dataset_id].class_names)
        out["networkid"] = networkid
        resp.text = json.dumps(out, indent=1, ensure_ascii=False)
        print("send NetworkArchitectureResource for network {networkid}")


class NetworkImageResourceResource:

    def on_get(self, req, resp, networkid : int, moduleid : int):
        network = networks[networkid]
        activation = network.get_activation(moduleid) # this is a cpu tensor
        # give first element of activations because we do not want to have the batch dimension
        activation = activation[0]
        tensor_to_uint_transform = utils.TransformToUint()
        data = tensor_to_uint_transform(activation, True)
        zero_value = tensor_to_uint_transform(0., False)
        
        # if the layer has 1D data, make a 2D image out of it
        if len(data.shape) == 1:
            data = data.reshape((-1, 1, 1))

        tensors = [utils.encode_image(ten) for ten in data]
        out = {
            "tensors" : tensors,
            "networkid" : networkid,
            "moduleid" : moduleid,
            "zeroValue" : int(zero_value),
            "mode" : "Activation",
            }
        resp.text = json.dumps(out, indent=1, ensure_ascii=False)
        print(f"send NetworkImageResourceResource for network {networkid} layer {moduleid}")


class NetworkFeatureVisualizationResource:

    def on_get(self, req, resp, networkid : int, moduleid : int):
        network = networks[networkid]
        data = network.get_feature_visualization(moduleid)
        if data.shape[1] == 3:
            data = data[:,np.array([2, 1, 0])] # for sorting color channels
        data = data.transpose([0, 2, 3, 1]) # put channel dimension to last
        tensors = [utils.encode_image(ten) for ten in data]
        out = {
            "tensors" : tensors,
            "networkid" : networkid,
            "moduleid" : moduleid,
            "mode" : "FeatureVisualization",
            }
        resp.text = json.dumps(out, indent=1, ensure_ascii=False)
        print(f"send NetworkFeatureVisualizationResource for network {networkid} layer {moduleid}")


class NetworkForwardPassResource:

    def on_put(self, req, resp, networkid : int):
        jsonfile = json.load(req.stream)
        image_resource = ImageResource(
            network_ID = jsonfile["networkID"],
            dataset_ID = jsonfile["datasetID"],
            image_ID = jsonfile["imageID"],
            module_ID = jsonfile["moduleID"],
            channel_ID = jsonfile["channelID"],
            noise_generator_ID = jsonfile["noiseGeneratorID"],
            mode = ImageResource.Mode(jsonfile["mode"]))
        
        network = networks[networkid]

        image = None
        if image_resource.mode == ImageResource.Mode.DatasetImage and image_resource.image_ID >= 0:
            image = datasets[image_resource.dataset_ID].get_data_item(image_resource.image_ID, True)[0]
        elif image_resource.mode == ImageResource.Mode.FeatureVisualization:
            image = networks[image_resource.network_ID].try_load_feature_visualization(image_resource.layer_ID)
            if image is not None:
                image = image[image_resource.channel_ID]
            else:
                raise falcon.HTTPBadRequest(title="Feature Visualization is not yet produced.")
        elif image_resource.mode == ImageResource.Mode.NoiseImage:
            image = noise_generators[image_resource.noise_generator_ID].get_noise_image()
        elif image_resource.mode == ImageResource.Mode.Activation:
            raise falcon.HTTPBadRequest(title="You cannot load an activation")
        else:
            image = torch.zeros(datasets[network.corresponding_dataset_id].get_data_item(0, True)[0].shape)
        image_resource.data = image

        network.forward_pass(image_resource)
        print(f"send NetworkPrepareForInputResource for network {networkid}")


class NetworkClassificationResultResource:

    def on_get(self, req, resp, networkid : int):
        network = networks[networkid]
        dataset = datasets[network.active_data_item.dataset_ID]
        class_names = dataset.class_names
        if not isinstance(class_names, np.ndarray):
            class_names = np.array(class_names) 
        posteriors, class_indices = network.get_classification_result()
        if posteriors is not None:
            out = {
                "networkid" : networkid,
                "class_names" : list(class_names[class_indices]), 
                "confidence_values" : [f"{item:.2f}" for item in posteriors],

            }
        else:
            out = {
                "networkid" : networkid,
                "class_names" : [], 
                "confidence_values" : [],
            }
        resp.text = json.dumps(out, indent=1, ensure_ascii=False)
        print(f"send NetworkClassificationResultResource for network {networkid}")
        



class NetworkWeightHistogramResource:

    def on_get(self, req, resp, networkid : int, moduleid : int):
        weights = networks[networkid].get_weights(moduleid)
        has_weights = weights is not None
        if has_weights:
            hist, bins = np.histogram(weights.detach().cpu().numpy(), 16)
            bins = 0.5 * (bins[1:] + bins[:-1])
        else:
            hist, bins = [], []
        out = {
            "networkid" : networkid,
            "moduleid" : moduleid,
            "has_weights" : str(has_weights),
            "counts" : [f"{item}" for item in hist], 
            "bins" : [f"{item}" for item in bins],
        }
        resp.text = json.dumps(out, indent=1, ensure_ascii=False)
        print(f"send NetworkWeightHistogramResource for network {networkid} layer {moduleid}")


class NetworkActivationHistogramResource:

    def on_get(self, req, resp, networkid : int, moduleid : int):
        activations = networks[networkid].get_activation(moduleid)
        hist, bins = np.histogram(activations.detach().cpu().numpy(), 16)
        bins = 0.5 * (bins[1:] + bins[:-1])
        out = {
            "networkid" : networkid,
            "moduleid" : moduleid,
            "counts" : [f"{item}" for item in hist], 
            "bins" : [f"{item}" for item in bins],
        }
        resp.text = json.dumps(out, indent=1, ensure_ascii=False)
        print(f"send NetworkActivationHistogramResource for network {networkid} module {moduleid}")

        
class DataImagesResource:

    def on_get(self, req, resp, datasetid : int):
        dataset = datasets[datasetid]
        out = dataset.get_data_overview()
        tensors = []
        labels = []
        for i in range(out["len"]):
            tensor, label = dataset.get_data_item(i, False)
            tensors.append(utils.tensor_to_string(tensor))
            labels.append(label)
        out['tensors'] = tensors
        out['label_ids'] = labels
        out["class_names"] = list(dataset.class_names)
        resp.text = json.dumps(out, indent=1, ensure_ascii=True)
        print("send DataImagesResource for dataset {datasetid}")


class DataNoiseImageResource:

    def on_get(self, req, resp, noiseid : int):
        noise_generator = noise_generators[noiseid]
        noise_generator.generate_noise_image()
        image = noise_generator.get_noise_image()
        image_enc = utils.tensor_to_string(image)
        out = {'tensor' : image_enc}
        resp.text = json.dumps(out, indent=1, ensure_ascii=False)
        print("send DataNoiseImageResource for noise generator {noiseid}")


class NetworkSetNetworkGenFeatVisResource:

    def on_put(self, req, resp, networkid : int):
        network = networks[networkid]
        network.feature_visualization_mode = FeatureVisualizationMode.Generating
        print(f"NetworkSetNetworkGenFeatVisResource for network {networkid}")


class NetworkSetNetworkLoadFeatVisResource:

    def on_put(self, req, resp, networkid : int):
        network = networks[networkid]
        network.feature_visualization_mode = FeatureVisualizationMode.Loading
        print(f"NetworkSetNetworkLoadFeatVisResource for network {networkid}")


class NetworkSetNetworkDeleteFeatVisResource:

    def on_put(self, req, resp, networkid : int):
        network = networks[networkid]
        network.delete_feat_vis_cache()
        print(f"NetworkSetNetworkDeleteFeatVisResource for network {networkid}")


class NetworkExportLayerResource:

    def on_put(self, req, resp, networkid : int):
        jsonfile = json.load(req.stream)
        image_resource = ImageResource(
            network_ID = jsonfile["networkID"],
            dataset_ID = jsonfile["datasetID"],
            image_ID = jsonfile["imageID"],
            layer_ID = jsonfile["layerID"],
            channel_ID = jsonfile["channelID"],
            noise_generator_ID = jsonfile["noiseGeneratorID"],
            mode = ImageResource.Mode(jsonfile["mode"]))
        
        network = networks[networkid]
        network.export(image_resource)
        print(f"send NetworkPrepareForInputResource for network {networkid}")
        

class TestShortResource:

    def on_get(self, req, resp):
        doc = {
            'Hello': [
                {
                    'World': 'Hip Huepf'
                }
            ]
        }
        resp.text = json.dumps(doc, ensure_ascii=False)


class TestLongResource:

    def on_get(self, req, resp):
        time.sleep(60)
        doc = {
            'Hello': [
                {
                    'World': 'Hip Huepf'
                }
            ]
        }
        resp.text = json.dumps(doc, ensure_ascii=False)"""