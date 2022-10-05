import falcon
from falcon import testing
import json
import pytest

import Scripts.DLWebServer as DLWebServer
import Scripts.Falconapp as falconapp
import run_server as run_server

# select the Python module, which defines what networks, datasets ect. should be loaded 
DLWebServer.select_project_server(run_server)
application = falconapp.start_server(DLWebServer)

@pytest.fixture
def client():
    return testing.TestClient(application)

# pytest will test all functions with the prefix "test_".
# pytest will inject the object returned by the "client" function as an additional parameter.
"""def test_list_images(client):
    doc = {
            'Hello': [
                {
                    'World': 'Hip Huepf'
                }
            ]
        }

    response = client.simulate_get('/testshortresource')
    result_doc = json.loads(response.content)

    assert result_doc == doc
    assert response.status == falcon.HTTP_OK"""

def test_get_architecture(client):
    response = client.simulate_get('/network/0')
    result_doc = json.loads(response.content)
    print(result_doc)

    assert response.status == falcon.HTTP_OK
    
"""
def test_get_activation(client):
    response = client.simulate_get('/network/0/activation/moduleid/1')
    result_doc = json.loads(response.content)
    print(result_doc)
    
    assert response.status == falcon.HTTP_OK
"""