#!/usr/bin/env python3

from typing import Any, Optional, Tuple

import faiss


import faiss.contrib.torch_utils


import torch


class KNN:

    def __init__(
        self,
        k: int = 1,
        metric: str = "l2",
        radius: Optional[float] = None,
        res: Optional[Any] = None,
    ) -> None:

        self.index: Any = None
        self.k: int = k
        self.metric: str = metric
        self.radius: Optional[float] = radius
        self.res: Optional[Any] = res

    def fit(self, data: torch.Tensor) -> None:

        dimensions = data.shape[1]

        if self.metric == "l2":
            self.index = faiss.IndexFlatL2(dimensions)
            if data.is_cuda:
                data = data.cpu()
            self.index.add(data)

        elif self.metric == "cosine":
            self.index = faiss.IndexFlatIP(dimensions)

            data = data / torch.linalg.norm(data, dim=1, keepdim=True)

            self.index.train(data)
            self.index.add(data)

        else:
            raise ValueError(f"Metric {self.metric} is not supported.")

    def search(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        is_cuda = False
        device = None
        if data.is_cuda:
            device = data.get_device()
            is_cuda = True
            data = data.cpu()

        if self.metric == "l2":
            if self.radius is None:
                distances, indices = self.index.search(data, k=self.k)
            else:

                distances, indices = self.index.range_search_with_radius(data, float(self.radius))

        elif self.metric == "cosine":

            data = data / torch.linalg.norm(data, dim=1, keepdim=True)

            similarity, indices = self.index.search(data, k=self.k)

            distances = 1.0 - similarity
        else:
            raise ValueError(f"Metric {self.metric} is not supported.")

        if is_cuda:
            distances = distances.to(device)
            indices = indices.to(device)

        return distances, indices

    def serialize_index(self) -> None:
        self.index = faiss.serialize_index(self.index)

    def deserialize_index(self) -> None:
        self.index = faiss.deserialize_index(self.index)
