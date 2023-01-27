import unittest


class TestPennFudan(unittest.TestCase):

    def test_file_to_list(self):
        # give
        import os
        root = './'
        path = 'detection'
        # then
        out = list(sorted(os.listdir(os.path.join(root, path))))
        print(f"test_file_to_list = {out}")
        # when
        self.assertIsNotNone(out)

    def test_path_to_file(self):
        import os
        # give
        root = './'
        path = 'detection'
        file = 'engine.py'
        # then
        out = os.path.join(root, path, file)
        print(f"test_path_to_file = {out}")
        # when
        self.assertIsNotNone(out)

    def output(self, params):
        import torch
        import datasets
        import detection.utils as utils

        dataset = datasets.PennFudan(
            params, datasets.get_transform(train=True))
        dataset_test = datasets.PennFudan(
            params, datasets.get_transform(train=False))

        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-50])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=utils.collate_fn)

        return data_loader, data_loader_test, dataset_test

    def test_output(self):
        # give
        params = {
            'root': 'PennFudanPed',
            'imgs_path': 'PNGImages',
            'masks_path': 'PedMasks'
        }
        # then
        data_loader, data_loader_test, dataset_test = self.output(
            params)

        print(f"test_output")
        # when
        self.assertIsNotNone(data_loader)
        self.assertIsNotNone(data_loader_test)
        self.assertIsNotNone(dataset_test)


if __name__ == '__main__':
    unittest.main()
