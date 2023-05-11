import asyncio
from Scripts.ImageResource import ImageResource
import websockets
import json
import os
import torch
import numpy as np
import Scripts.utils as utils
from Scripts.utils import get_module

# Create global variables. They will be filled inplace.
network = None
dataset = None
input_shape = None
transform_norm = None
noise_generator = None
current_image_resource = None
current_fv_image_resource = None
loader_test_norm = None
pending_data_to_send = []
lock = asyncio.Lock()

def prepare_dl_objects(source):
    global dataset
    global transform_norm
    global loader_test_norm
    dataset, transform_norm, loader_test_norm = source.get_dataset()
    global network
    network = source.get_network(report_feature_visualization_results)
    # Set the class names for the network. This is important if class names should be displayed for some channels.
    network.class_names = dataset.class_names
    global noise_generator
    noise_generator = source.get_noise_generator()

    global input_shape
    input_shape = dataset.get_data_item(0)[0].shape
    data = torch.zeros(input_shape)
    image_resource = ImageResource(id=0, mode=ImageResource.Mode.DATASET, data=data)
    network.initial_forward_pass(image_resource)


async def handler(websocket):

    # create periodic task:
    asyncio.create_task(send(websocket))

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
            response = await request_image_data(event)
        elif event["resource"] == "set_network_weights":
            response = set_network_weights(event)
        elif event["resource"] == "set_fv_image_resource":
            response = set_fv_image_resource(event)
        elif event["resource"] == "request_noise_image":
            response = request_noise_image(event)
        elif event["resource"] == "set_fv_settings":
            response = set_fv_settings(event)
        elif event["resource"] == "request_test_results":
            response = request_test_results(event)
        
        if response:
            await websocket.send(response)


async def send(websocket):
    while True:
        async with lock:
            while pending_data_to_send:
                item = pending_data_to_send.pop()
                print("send pending data")
                await websocket.send(item)
        await asyncio.sleep(0.5)


def request_dataset_images(event):
    image_resources = []
    n = event["n"]
    if n > 0:
        rand_inds = np.random.randint(len(dataset), size=n)
        for rand_ind in rand_inds:
            tensor, label = dataset.get_data_item(rand_ind)
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
    image_resource = get_image_resource(event["image_resource"])
    response = perform_forward_pass(image_resource)
    print("send: " + "request_forward_pass")
    return response


def perform_forward_pass(image_resource=None):
    # Testing end
    global current_image_resource
    if image_resource is None and current_image_resource is not None:
        image_resource = current_image_resource
    elif image_resource is not None:
        current_image_resource = image_resource
    else:
        return ""
    
    # Perform the forward pass
    global network
    global transform_norm
    data = transform_norm(image_resource.data)
    network.forward_pass(data)
    
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
    return response


def request_architecture(event):
    architecture_dict = network.get_architecture()
    response = {"resource" : "request_architecture", "architecture" : architecture_dict}
    response = json.dumps(response, indent=1, ensure_ascii=False)
    print("send: " + "request_architecture")
    return response


async def request_image_data(event):
    print(event)
    image_resources = []
    module_resource = event["network_module_resource"]
    mode = event["mode"]
    module_id = module_resource["module_id"]
    channel_labels = network.get_channel_labels(module_id)
    if mode == "activation":
        activations = network.get_activation(module_id)[0]
        activations, minimum, maximum = utils.normalize(activations)
        if activations.ndim == 1:
            # activations should have shape [C, H, W]
            # The activation was flattend. We have to convert it to C, H, W format.
            activations = activations.unsqueeze(1).unsqueeze(1)
        for channel_id, activation in enumerate(activations):
            image_resources.append(ImageResource(
                module_id=module_id,
                channel_id=channel_id,
                mode=ImageResource.Mode.ACTIVATION,
                label=channel_labels[channel_id] if channel_labels else "",
                data=utils.tensor_to_string(activation.unsqueeze(0)), # convert activation shape to [C, H, W] with C = 1
                value_zero_decoded=minimum.item(),
                value_255_decoded=maximum.item(),
            ).__dict__)
    elif mode == "fv":
        if current_fv_image_resource is None:
            print("FV request cannot be served: no fv image resource selected.")
            return ""
        asyncio.create_task(network.compute_feature_visualization(module_id, transform_norm(current_fv_image_resource.data)))
        image_resources = []
    else:
        raise ValueError(f"Unknown mode {mode}")
    response = {"resource" : "request_image_data", "image_resources" : image_resources}
    response = json.dumps(response, indent=1, ensure_ascii=True)
    print("send: " + "request_image_data")
    return response


def set_network_weights(event):
    for module_id, data in event['weight_dicts'].items():
        for weight_name, weights in data.items():
            weights = torch.FloatTensor(weights)
            weights = weights.to(network.device)
            network.set_data(int(module_id), weight_name, weights)
    response = perform_forward_pass()
    print("send: " + "set_network_weights, request_forward_pass")
    return response


def set_fv_image_resource(event):
    image_resource = get_image_resource(event["image_resource"])
    global current_fv_image_resource
    current_fv_image_resource = image_resource
    print("set_fv_image_resource")
    return ""


def set_fv_settings(event):
    fv_settings_dict = event["fv_settings"]
    global network
    network.set_feature_visualization_params(fv_settings_dict)
    print("set_fv_settings")
    return ""


async def report_feature_visualization_results(images, channels, module_id):
    channel_labels = network.get_channel_labels(module_id)
    
    image_resources = []

    for channel_id, image in zip(channels, images):
        image_resources.append(ImageResource(
            module_id=module_id,
            channel_id=int(channel_id),
            mode=ImageResource.Mode.FEATURE_VISUALIZATION,
            label=channel_labels[channel_id] if channel_labels else "",
            data=utils.tensor_to_string(image),
        ).__dict__)

    response = {"resource" : "request_image_data", "image_resources" : image_resources}
    response = json.dumps(response, indent=1, ensure_ascii=True)
    async with lock:
        pending_data_to_send.append(response)
    # Switch to other tasks (sending data)
    await asyncio.sleep(0.5)
    


def get_image_resource(image_resource_dict):
    # Create image resource
    image_resource = ImageResource(
        id = image_resource_dict["id"],
        module_id = image_resource_dict["module_id"],
        channel_id = image_resource_dict["channel_id"],
        mode = ImageResource.Mode(image_resource_dict["mode"]),
    )

    # Create image for the image resource
    image = None
    global input_shape
    if "data" in image_resource_dict:
        image = utils.decode_image(image_resource_dict["data"], input_shape[0])
    else:
        image = torch.zeros(input_shape)
    image_resource.data = image

    return image_resource


def request_noise_image(event):
    noise_image = noise_generator.get_noise_image()
    image_resource = ImageResource(
        mode = ImageResource.Mode.NOISE,
        data = utils.tensor_to_string(noise_image)
    )
    image_resource = image_resource.__dict__
    response = {"resource" : "request_noise_image", "image_resource" : image_resource}
    response = json.dumps(response, indent=1, ensure_ascii=True)
    print("send: " + "request_noise_image")
    return response


def request_test_results(event):
    
    targets = []
    preds = []
    n_images = 0
    global loader_test_norm

    with torch.no_grad():
        for data_ in loader_test_norm:
            target = data_['label']
            data = data_['data']
            targets.append(target.cpu().numpy())
            n_images += len(target)
            data = data.to(network.device)
            outputs = network.model({"data" : data})
            pred = outputs['logits'].argmax(1)
            preds.append(pred.cpu().numpy())
        
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    accuracy = (preds == targets).sum() / n_images
    fig, _ = utils.draw_confusion_matrix(targets, preds, dataset.class_names)
    encoded_image = utils.get_image_from_fig(fig)
    
    response = {"resource" : "request_test_results", "accuracy" : accuracy, "conf_matrix" : encoded_image}
    response = json.dumps(response, indent=1, ensure_ascii=True)
    print("send: " + "request_noise_image")
    return response



async def main():
    async with websockets.serve(handler, "", 8000, max_size=2**30, read_limit=2**30, write_limit=2**30):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    project = "HFNetMNIST"
    #project = "Flowers102"
    source_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "Projects", project, "get_dl_objects.py")
    get_dl_objects_module = get_module(source_path)
    prepare_dl_objects(get_dl_objects_module)
    print("server running")
    asyncio.run(main())