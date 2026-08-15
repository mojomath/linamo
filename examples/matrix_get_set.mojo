from linamo.prelude import *


def main() raises:
    var m = la.matrix[int64](
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
            [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
            [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
            [61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
            [71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
        ]
    )
    m[0, 0] = 999
    print("A 8x10 matrix:")
    print(m)

    var v1 = m[1::2, 0:8:3]
    print("A view of the matrix. Rows 1, 3, 5, 7. Columns 0, 3, 6:")
    print(v1)

    var v2 = v1[1::2, 1::]
    print("A view of the previous view. Rows 1, 3. Columns 1, 2:")
    print(v2)

    print("The first element of v1:", v1[0, 0])
    print("The first element of v2:", v2[0, 0])
