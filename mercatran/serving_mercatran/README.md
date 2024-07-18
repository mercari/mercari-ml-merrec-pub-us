# Serving MercaTran using Torchserve
To serve MercaTran, we create custom containers running Torchserve model server
which can be deployed on public cloud platforms like Vertex AI. 
Our framework embeds both users and items, hence we create two containers for MercaTran serving. 

### Create a model that can be deployed with Torchserve
To create a model that can run in a Torch model server, it needs:

1. A custom handler, which needs to inherit
from the [base handler](https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py).
The default [text](https://github.com/pytorch/serve/blob/master/ts/torch_handler/text_handler.py) and 
[vision handlers](https://github.com/pytorch/serve/blob/master/ts/torch_handler/vision_handler.py) provided in Torchserve can also be used for simpler applications. For this project, we create a base custom handler (`merrec_handler_base.py`) which is used as a base class for both user and item models.

2. Model files (`.pth`) and other files like (`vocab.pt`) (`tokenizer.json` in our case), etc.

The following command can be used to create a model archive for Torchserve. In this project, however, we perform
this action in the docker container itself. Refer to [Torchserve Examples](https://github.com/pytorch/serve/tree/master/examples/text_classification) for full details.

```bash
torch-model-archiver --model-name {end_point_name} --version 1.0 --model-file model.py --serialized-file model.pt --handler text_classifier --extra-files "post_process.json,source_vocab.pt"
```
The `--extra-files` option can be used to add any other dependencies like a file that stores class index to name
mapping for easier understanding of the predictions or the vocab file used to train the model.

To deploy the models, the service needs a configuration file (see `config.properites`)
which stores the default ip-address and ports to serve requests.

## Create custom docker container running Torserve.

In this project, we create separate containers for user end-point and item end-point. Since
these share the model and a signifcant portion of the code, we create a class hierarchy and supply
the base classes using `--extra-files`. This is a workaround, and ideally should be a python package
that can be installed and imported.

To create the user container, make sure `model.pt` and `tokenizer.json` (the trained tokenizer file refer to training pipeline) 
are in the current directory along with the other required files (see `Dockerfile.user` and `Dockerfile.item`) and run
the following command.

```bash
sudo docker build --tag="mercatran-user:gpu" -f Dockerfile.user .
```
Create the item container using this command:

```bash
sudo docker build --tag="mercatran-item:gpu" -f Dockerfile.item .
```
## Run the container locally

The built containers can be run locally using the following commands:
Run the user container
```bash
docker run -t -d --rm -p 7080:7080 mercatran-user:gpu 
```
Note: To run both containers locally, make sure in the `config.properties` to set the ports differently
to avoid collisions.


### Running inference locally
Here are sample scripts to run inference on user endpoint. Note: See the sample data format
expected by each container. 

Note: The model in its current config supports a batch size of 128, this can be changed in the `model_config.py`. 
Requests smaller than 128, are padded server side to run the inference, however the server will return 
the results for only the original request.
Note: The embedding dimension is 64 currently. To change this, the model will need to be re-trained with a new embedding size.
Refer to `D_MODEL` in `config.py` of the training pipeline.

User Portion of MercaTran
```python
import requests
# Sample request, batch size = 3
# Note: Seq len for each request is different
# Note: Seq len for any request shouldn't exceed 22

data_user = {
    "instances": [
        {
            "user": [
                {
                    "title": [
                        "Louis Vuitton Lock and Key Brass 313",
                        "Suncloud, polarized, reader, sunglasses",
                        "Rawlings 9inch SOFTBALL Left",
                    ],
                    "brand": ["Louis Vetton", "SunCloud Polarized Optics", "Rawlings"],
                    "category": [
                        "Vingtage & Collectibles",
                        "Sunglasses SunCloud Polarized Optics Sunglasses for Men",
                        "Baseball Equipment",
                    ],
                },
                {
                    "title": ["  ", "*&(*&)( .   "],
                    "brand": [":)^&**&*", ""],
                    "category": ["Shoes", ""],
                },
                {
                    "title": [None, "test"],
                    "brand": [None, "no test"],
                    "category": [None, "yes test"],
                },
            ]
        }
    ]
}

r = requests.post(
    "http://0.0.0.0:7080/predictions/mercatran_user", json=data_user)
timesteps = r.json()["predictions"][0]
# user endpoint has 4 timesteps
for timestep in timesteps:
    print("TimeStep: ", timestep)
    print("Embedding: ", timesteps[timestep])
```

Stop the user container.
Run the item container
```bash
docker run -t -d --rm -p 7080:7080 mercatran-item:gpu 
```

Item Portion of MercaTran
```python

import requests

# Sample Request, batch size = 3
# Note that seq len for each request (i.e each datapoint in the batch should be just one item)
data_item = {
    "instances": [
        {
            "item": [
                {
                    "title": ["Louis Vuitton Lock and Key Brass 313"],
                    "brand": ["Louis Vetton"],
                    "category": ["Vingtage & Collectibles"],
                },
                {
                    "title": ["*&(*&)( .   "],
                    "brand": [""],
                    "category": ["Shoes"],
                },
                {
                    "title": ["Liverpool Jersey"],
                    "brand": ["Nike"],
                    "category": ["Apparel"],
                },
                {"title": [None], "brand": [None], "category": [None]},
            ]
        }
    ]
}

r = requests.post(
    "http://0.0.0.0:7080/predictions/mercatran_item", json=data_item)
timesteps = r.json()["predictions"][0]
# item endoint has 1 timestep
for timestep in timesteps:
    print("TimeStep: ", timestep)
    print("Embedding: ", timesteps[timestep])
```
