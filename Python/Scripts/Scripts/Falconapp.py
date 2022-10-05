import falcon

def start_server(DLWebServer):
    
    app = application = falcon.App()

    app.add_route('/testshortresource', DLWebServer.TestShortResource())
    app.add_route('/testlongresource', DLWebServer.TestLongResource())

    app.add_route('/network', DLWebServer.NetworkResource())
    app.add_route('/network/{networkid:int}', DLWebServer.NetworkArchitectureResource())
    app.add_route('/network/{networkid:int}/activation/moduleid/{moduleid:int}', DLWebServer.NetworkActivationImageResource())
    app.add_route('/network/{networkid:int}/featurevisualization/moduleid/{moduleid:int}', DLWebServer.NetworkFeatureVisualizationResource())
    app.add_route('/network/{networkid:int}/forwardpass', DLWebServer.NetworkForwardPassResource())
    app.add_route('/network/{networkid:int}/classificationresult', DLWebServer.NetworkClassificationResultResource())
    app.add_route('/network/{networkid:int}/weighthistogram/moduleid/{moduleid:int}', DLWebServer.NetworkWeightHistogramResource())
    app.add_route('/network/{networkid:int}/activationhistogram/moduleid/{moduleid:int}', DLWebServer.NetworkActivationHistogramResource())

    app.add_route('/network/{networkid:int}/setnetworkgenfeatvis', DLWebServer.NetworkSetNetworkGenFeatVisResource())
    app.add_route('/network/{networkid:int}/setnetworkloadfeatvis', DLWebServer.NetworkSetNetworkLoadFeatVisResource())
    app.add_route('/network/{networkid:int}/setnetworkdeletefeatvis', DLWebServer.NetworkSetNetworkDeleteFeatVisResource())

    app.add_route('/network/{networkid:int}/export', DLWebServer.NetworkExportLayerResource())

    app.add_route('/dataset/{datasetid:int}/images', DLWebServer.DataImagesResource())

    app.add_route('/noiseimage/{noiseid:int}', DLWebServer.DataNoiseImageResource())

    return application