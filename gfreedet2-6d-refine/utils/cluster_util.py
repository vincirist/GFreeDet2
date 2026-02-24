# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

from typing import Tuple

import faiss
import torch
from utils import torch_helpers


def kmeans(
    samples: torch.Tensor,
    num_centroids: int,
    num_iter: int = 50,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    gpu = True if samples.device.type == "cuda" else False

    num_dims = samples.shape[1]
    kmeans = faiss.Kmeans(
        num_dims,
        num_centroids,
        niter=num_iter,
        gpu=gpu,
        verbose=True,
        seed=0,
        spherical=False,
    )

    samples_cpu = samples.cpu().numpy()

    kmeans.train(samples_cpu)

    centroid_distances, cluster_ids = kmeans.index.search(samples_cpu, 1)

    centroid_distances = centroid_distances.squeeze(axis=-1)
    cluster_ids = cluster_ids.squeeze(axis=-1)

    device = samples.device
    return (
        torch_helpers.array_to_tensor(kmeans.centroids).to(device),
        torch_helpers.array_to_tensor(cluster_ids).to(torch.int32).to(device),
        torch_helpers.array_to_tensor(centroid_distances).to(device),
    )
