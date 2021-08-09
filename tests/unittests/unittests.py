from utils import *
from unittest import TestCase

"""
- For each operation, you should write tests to test  on matrices of different sizes.
- Keep in mind that the tests provided in the starter code are NOT comprehensive. That is, we strongly
advise you to modify them and add new tests.
- Hint: use dp_mc_matrix to generate dumbpy and numc matrices with the same data and use
      cmp_dp_nc_matrix to compare the results
"""

small_rows = 10**1 + 9
small_cols = 10**1 + 9
medium_rows = 2 * (10**2) + 9
medium_cols = 2 * (10**2) + 9
high_rows = (10**3)
high_cols = (10**3)

class TestAdd(TestCase):
    def test_small_numbers_add(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(medium_rows, medium_rows,low=-1, high=1, seed=1)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(medium_rows, medium_rows,low=-1, high=1, seed=19)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "add")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_small_add(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(small_rows, small_cols, seed=11)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(small_rows, small_cols, seed=123)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "add")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_add(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(medium_rows, medium_cols, seed=1467)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(medium_rows, medium_cols, seed=159)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "add")
        self.assertTrue(is_correct)
        print_speedup(speed_up)


    def test_large_add(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(high_rows, high_cols, seed=189)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(high_rows, high_cols, seed=1123)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "add")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

# (OPTIONAL) Uncomment the following TestSub class if you have implemented matrix subtraction.
# class TestSub(TestCase):
#    def test_small_sub(self):
#        # TODO: YOUR CODE HERE
#        dp_mat1, nc_mat1 = rand_dp_nc_matrix(2, 2, seed=0)
#        dp_mat2, nc_mat2 = rand_dp_nc_matrix(2, 2, seed=1)
#        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "sub")
#        self.assertTrue(is_correct)
#        try:
#            nc.Matrix(3, 3) - nc.Matrix(2, 2)
#            self.assertTrue(False)
#        except ValueError as e:
#            print(e)
#            pass
#        print_speedup(speed_up)
#
#    def test_medium_sub(self):
#        # TODO: YOUR CODE HERE
#        pass
#
#    def test_large_sub(self):
#        # TODO: YOUR CODE HERE
#        pass

class TestAbs(TestCase):
    def test_small_numbers_abs(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(medium_rows, medium_cols, low=-1, high=1, seed=71)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "abs")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_small_abs(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(small_rows, small_cols, seed=634)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "abs")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_abs(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(medium_rows, medium_cols, seed=71)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "abs")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_abs(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(high_rows, high_cols, seed=92)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "abs")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

# (OPTIONAL) Uncomment the following TestNeg class if you have implemented matrix negation.
# class TestNeg(TestCase):
#    def test_small_neg(self):
#        # TODO: YOUR CODE HERE
#        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
#        is_correct, speed_up = compute([dp_mat], [nc_mat], "neg")
#        self.assertTrue(is_correct)
#        print_speedup(speed_up)
#    def test_medium_neg(self):
#        # TODO: YOUR CODE HERE
#        pass

#    def test_large_neg(self):
#        # TODO: YOUR CODE HERE
#        pass

class TestMul(TestCase):
    def test_small_numbers_mul(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(medium_rows, medium_rows,low=-1, high=1, seed=1)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(medium_rows, medium_rows,low=-1, high=1, seed=19)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_small_mul(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(small_rows, small_cols, seed=11)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(small_rows, small_cols, seed=123)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_mul(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(medium_rows, medium_cols, seed=1467)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(medium_rows, medium_cols, seed=159)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
        self.assertTrue(is_correct)
        print_speedup(speed_up)


    def test_large_mul(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(high_rows, high_cols, seed=189)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(high_rows, high_cols, seed=1123)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
        self.assertTrue(is_correct)
        print_speedup(speed_up)



class TestPow(TestCase):
    def test_small_pow(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(small_rows, small_cols, seed=1323)
        is_correct, speed_up = compute([dp_mat, 3], [nc_mat, 3], "pow")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_pow(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(medium_rows, medium_cols, seed=675)
        is_correct, speed_up = compute([dp_mat, 33], [nc_mat, 33], "pow")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_pow(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(high_rows, high_cols, seed=1876)
        is_correct, speed_up = compute([dp_mat, 120], [nc_mat, 120], "pow")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

class TestGet(TestCase):
    def test_get(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        rand_row = np.random.randint(dp_mat.shape[0])
        rand_col = np.random.randint(dp_mat.shape[1])
        self.assertEqual(round(dp_mat.get(rand_row, rand_col), decimal_places),
            round(nc_mat.get(rand_row, rand_col), decimal_places))

class TestSet(TestCase):
    def test_set(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        rand_row = np.random.randint(dp_mat.shape[0])
        rand_col = np.random.randint(dp_mat.shape[1])
        dp_mat.set(rand_row, rand_col, 2)
        nc_mat.set(rand_row, rand_col, 2)
        self.assertTrue(cmp_dp_nc_matrix(dp_mat, nc_mat))

class TestShape(TestCase):
    def test_shape(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        self.assertTrue(dp_mat.shape == nc_mat.shape)

class TestIndexGet(TestCase):
    def test_index_get(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        rand_row = np.random.randint(dp_mat.shape[0])
        rand_col = np.random.randint(dp_mat.shape[1])
        self.assertEqual(round(dp_mat[rand_row][rand_col], decimal_places),
            round(nc_mat[rand_row][rand_col], decimal_places))

class TestIndexSet(TestCase):
    def test_set(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        rand_row = np.random.randint(dp_mat.shape[0])
        rand_col = np.random.randint(dp_mat.shape[1])
        dp_mat[rand_row][rand_col] = 2
        nc_mat[rand_row][rand_col] = 2
        self.assertTrue(cmp_dp_nc_matrix(dp_mat, nc_mat))
        self.assertEqual(nc_mat[rand_row][rand_col], 2)

class TestSlice(TestCase):
    def test_slice(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        self.assertTrue(cmp_dp_nc_matrix(dp_mat[0], nc_mat[0]))
        self.assertTrue(cmp_dp_nc_matrix(dp_mat[1], nc_mat[1]))
