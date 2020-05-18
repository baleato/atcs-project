import unittest
import torch
import math
import os
import numpy as np

from models import *
from util import *
from tasks import OffensevalTask

class TestModels(unittest.TestCase):

    def setUp(self):
        self.args = get_args_meta()
        self.model_path = '/tmp/model_test.pkl'
        self.device = get_pytorch_device(self.args)

    def check_models(self, model1, model2):
        state_dict_1 = model1.state_dict()
        state_dict_2 = model2.state_dict()
        for k1, k2 in zip(state_dict_1, state_dict_2):
            self.assertEqual(k1, k2)
            array1 = state_dict_1[k1].detach().numpy()
            array2 = state_dict_2[k2].detach().numpy()
            np.testing.assert_allclose(array1, array2)

    def print_model_size(self, model):
        print('{} - {}MB'.format(
            model.__class__.__name__,
            os.stat(self.model_path).st_size >> 20))

    def test_MetaLearner_save_load(self):
        task = OffensevalTask()
        args = self.args
        model = MetaLearner(args)
        model.add_task_classifier(task.get_name(), task.get_classifier().to(self.device))
        model.save_model(args.unfreeze_num, self.model_path)
        self.print_model_size(model)

        model2 = MetaLearner(args)
        model2.load_model(self.model_path, self.device)
        model2.add_task_classifier(task.get_name(), task.get_classifier().to(self.device))

        self.check_models(model, model2)

    def test_ProtoMAMLLearner_save_load(self):
        model = ProtoMAMLLearner(self.args, hidden_dims=[500])
        model.save_model(self.args.unfreeze_num, self.model_path)
        self.print_model_size(model)
        model2 = ProtoMAMLLearner(self.args, hidden_dims=[500])
        model2.load_model(self.model_path, self.device)
        self.check_models(model, model2)


if __name__ == '__main__':
    unittest.main()
