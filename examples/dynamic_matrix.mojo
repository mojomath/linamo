from linamo.prelude import *


def main() raises:
    # creation()
    calculation()


def creation() raises:
    var mat1 = la.matrix[float64](
        [
            [1.1, 1.2, 1.3, 1.4],
            [2.1, 2.2, 2.3, 2.4],
            [3.1, 3.2, 3.3, 3.4],
            [4.1, 4.2, 4.3, 4.4],
            [5.1, 5.2, 5.3, 5.4],
        ],
        order="C",
    )
    print("Memory layout (row-major):\n", mat1.data)

    var mat2 = la.matrix[float64](
        [
            [1.1, 1.2, 1.3, 1.4],
            [2.1, 2.2, 2.3, 2.4],
            [3.1, 3.2, 3.3, 3.4],
            [4.1, 4.2, 4.3, 4.4],
            [5.1, 5.2, 5.3, 5.4],
        ],
        order="F",
    )
    print("Memory layout (column-major):\n", mat2.data)

    var mat3 = la.matrix[int64](
        flat_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        nrows=3,
        ncols=4,
        order="C",
    )
    print("Matrix (row-major with shape):\n", mat3)

    var mat4 = la.matrix[int64](
        flat_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        nrows=3,
        ncols=4,
        order="F",
    )
    print("Matrix (column-major with shape):\n", mat4)


def calculation() raises:
    var a = la.matrix[float64](
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        order="C",
    )
    var b = la.matrix[float64](
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        order="C",
    )
    var c = a.copy()
    for _ in range(10000):
        c = c @ b
    print("Result of matrix multiplication:\n", c)
