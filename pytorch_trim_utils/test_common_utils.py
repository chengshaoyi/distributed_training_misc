import unittest
import numpy as np

from pytorch_trim_utils.common_utils import generate_trimed_weight_at_axis

class TestNumpyTrim(unittest.TestCase):
    arr_4x10 = np.arange(0,80).reshape([8,10])
    def test_trim_single(self):
        print("trim_single")
        print(self.arr_4x10)
        filter_indices_for_removal = [2, 4, 5, 7]
        trimmed = generate_trimed_weight_at_axis(self.arr_4x10, filter_indices_for_removal,0,1)
        print(trimmed)
        print(trimmed.shape)
    def test_trim_inner(self):
        print("trim_inner")
        print(self.arr_4x10)
        filter_indices_for_removal = [1, 3]
        trimmed = generate_trimed_weight_at_axis(self.arr_4x10, filter_indices_for_removal,1, 1)
        print(trimmed)
    def test_trim_multi(self):
        print("trim_multi")
        print(self.arr_4x10)
        filter_indices_for_removal = [1, 3]
        trimmed = generate_trimed_weight_at_axis(self.arr_4x10, filter_indices_for_removal,0, 2)
        print(trimmed)

    def test_trim_multi(self):
        print("trim_multi_inner")
        print(self.arr_4x10)
        filter_indices_for_removal = [1, 3]
        trimmed = generate_trimed_weight_at_axis(self.arr_4x10, filter_indices_for_removal,1, 2)
        print(trimmed)

'''def main():
    print("test")'''
'''
    old_weight_0 = np.arange(0,10).reshape([1,10])
    old_weight_1 = np.arange(0,10).reshape([1,10])
    old_weight = np.concatenate((old_weight_0,old_weight_1))
    old_weight = np.transpose(old_weight,[1,0])
    print(old_weight)
    print(old_weight.shape)
    print("-----------")
    new_weight = generate_trimed_weight_outer(old_weight, filter_indices_for_removal)
    print(new_weight)
    print(new_weight.shape)
    print("============")'''


'''
    print(old_weight)
    filter_indices_for_removal = [2, 3]
    new_weight = generate_trimed_weight_at_axis(old_weight, filter_indices_for_removal)
    print(new_weight)
    # old_weight_tr = np.transpose(old_weight,[1,0])
    print(generate_trimed_weight_at_axis(old_weight, filter_indices_for_removal, 1))'''


if __name__ == '__main__':
    unittest.main()
