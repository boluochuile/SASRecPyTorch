import time
import torch
from cluster.kmeans import kmeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # 创建数据集
    X, y = make_blobs(n_samples=200, n_features=2, centers=3, random_state=1)

    # Half type to save more GPU memory.
    pt_whitened = torch.from_numpy(X).half().cuda()
    pt_whitened = pt_whitened.unsqueeze(0)
    pt_whitened = pt_whitened.repeat([3, 1, 1])
    # kmeans
    torch.cuda.synchronize()
    s = time.time()
    for i in range(3):
        # best_centers, best_distance
        codebook, distortion = kmeans(pt_whitened[i], 4, batch_size=200, iter=5)
        print(codebook)
        torch.cuda.synchronize()
        e = time.time()
        print("Time: ", e-s)

        # Show
        plt.scatter(X[:, 0], X[:, 1], marker='o', s=8)
        plt.scatter(codebook.cpu()[:, 0], codebook.cpu()[:, 1], c='r')
        plt.show()
