from pics_extractor import MercadoLibre
from neural_network import Model

if __name__ == '__main__':
    extractor = MercadoLibre()
    extractor.get_imgs_motorcycle()
    extractor.get_imgs_bike()
    extractor.creating_testing_dataset()

    nn = Model()

    if nn.load_model():
        nn.predict()
    else:
        nn.define_generators()
        nn.create_architecture()
        nn.training()
        nn.evaluation()
        nn.save_model()
