import time
import numpy as np


# m, n, k in [512, 2048]
def getRandomMat(m, n, k):
    mat1 = np.random.randint(1, 100, (m, n)).tolist()
    mat2 = np.random.randint(1, 100, (n, k)).tolist()
    
    return mat1, mat2

def main():
    tmp = input('input m, n, k:\n')
    nums = tmp.split(' ')
    m, n, k = int(nums[0]), int(nums[1]), int(nums[2])
    mat1, mat2 = getRandomMat(m, n, k)

    ans = np.zeros((m, k)).tolist()
    start = time.time()
    for i in range(m):
        mat1_row = mat1[i][:]
        for j in range(k):
            mat2_col = mat2[:][j]
            ans[i][j] = sum([ x*y for x,y in zip(mat1_row, mat2_col)])

    end = time.time()
    print(f'running time: {end - start:.4f} s')
    #for a in ans:
    #    print(a)



if __name__ == '__main__':
    main()
    