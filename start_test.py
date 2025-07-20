from preprocessing import lincs_test_preprocessing
from train import test_perturbation_model
from utils import load_UniCure_pretrained_model


def test_lincs(seed):
    test_loader, lincs_gene = lincs_test_preprocessing(seed)
    unicure_model = load_UniCure_pretrained_model(path=f'./requirement/model_weights/best_model.pth')
    test_perturbation_model(unicure_model, test_loader, seed, lincs_gene, dataset_name='lincs2020', max_size=10)


if __name__ == '__main__':

    test_lincs(3)
