import torch
import numpy as np
from numpy import inf
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering, KMeans

from itertools import combinations

class Aggregator:
    def __init__(self, method: str) -> None:
        self.method = method
        self.func = getattr(self, f"_{self.method}", None)
        if self.func is None:
            raise ValueError(f"Method {self.method} is not defined")
    
    def __call__(self, all_gradients: torch.Tensor, *args, **kwargs) -> np.ndarray:
        """all_gradients: shape of (num, length of gradient)
        return an adversary gradient in shape of (1, length of gradient)"""
        res = self.func(all_gradients, *args, **kwargs).cpu()
        return res.numpy()
    
    @staticmethod
    def _pairwise_euclidean_distances(vectors):
        """Compute the pairwise euclidean distance.

        Arguments:
            vectors {list} -- A list of vectors.

        Returns:
            dict -- A dict of dict of distances {i:{j:distance}}
        """
        n = len(vectors)
        vectors = [v.flatten() for v in vectors]

        distances = {}
        for i in range(n - 1):
            distances[i] = {}
            for j in range(i + 1, n):
                distances[i][j] = ((vectors[i] - vectors[j]).norm()) ** 2
        return distances

    @staticmethod
    def _multi_krum_selection(distances, n, f, m):
        """Multi_Krum algorithm.

        Arguments:
            distances {dict} -- A dict of dict of distance. distances[i][j] = dist.
            i, j starts with 0.
            n {int} -- Total number of workers.
            f {int} -- Total number of excluded workers.
            m {int} -- Number of workers for aggregation.

        Returns:
            list -- A list indices of worker indices for aggregation. length <= m
        """
        def _compute_scores(distances, i, n, f):
            """Compute scores for node i.

            Args:
                distances {dict} -- A dict of dict of distance. distances[i][j] = dist.
                i, j starts with 0.
                i {int} -- index of worker, starting from 0.
                n {int} -- total number of workers
                f {int} -- Total number of excluded workers.

            Returns:
                float -- krum distance score of i.
            """
            s = [distances[j][i] ** 2 for j in range(i)] + [
                distances[i][j] ** 2 for j in range(i + 1, n)
            ]
            _s = sorted(s)[: n - f - 2]
            return sum(_s)

        if 2 * f + 2 > n:
            raise ValueError("Too many excluded workers: 2 * {} + 2 >= {}.".format(f, n))

        for i in range(n - 1):
            for j in range(i + 1, n):
                if distances[i][j] < 0:
                    raise ValueError(
                        "The distance between node {} and {} should be non-negative: "
                        "Got {}.".format(i, j, distances[i][j])
                    )

        scores = [(i, _compute_scores(distances, i, n, f)) for i in range(n)]
        sorted_scores = sorted(scores, key=lambda x: x[1])
        return list(map(lambda x: x[0], sorted_scores))[:m]
    
    def _mean(self, params: torch.Tensor, *args, **kwargs):
        return params.mean(dim=0)

    def _median(self, params: torch.Tensor, *args, **kwargs):
        values_upper, _ = params.median(dim=0)
        values_lower, _ = (-params).median(dim=0)
        res = (values_upper - values_lower) / 2
        return res
    
    def _multikrum(
        self,
        params: torch.Tensor,
        num_excluded: int = 1,
        num_aggregation: int = 1,
        *args, **kwargs
    ):
        distances = self._pairwise_euclidean_distances(params)
        top_m_indices = self._multi_krum_selection(
            distances, len(params), num_excluded, num_aggregation
        )
        values = torch.stack(
            [params[i] for i in top_m_indices], dim=0
        ).mean(dim=0)
        return values

    def _clustering(
        self,
        params: torch.Tensor,
        *args, **kwargs,
    ):
        num = len(params)
        dis_max = np.zeros((num, num))
        for i in range(num):
            for j in range(i + 1, num):
                dis_max[i, j] = 1 - F.cosine_similarity(
                    params[i, :], params[j, :], dim=0
                )
                dis_max[j, i] = dis_max[i, j]
        dis_max[dis_max == -inf] = 0
        dis_max[dis_max == inf] = 2
        dis_max[np.isnan(dis_max)] = 2
        clustering = AgglomerativeClustering(
            affinity="precomputed", linkage="average", n_clusters=2
        )
        clustering.fit(dis_max)
        flag = 1 if np.sum(clustering.labels_) > num // 2 else 0
        selected_idx = [idx for idx, label in enumerate(clustering.labels_) if label == flag]
        values = params[selected_idx].mean(dim=0)
        return values

    def _signguard(
        self,
        params: torch.Tensor,
        agg="mean",
        linkage="average",
        *args, **kwargs,
    ):
        assert linkage in ["average", "single"]
        num = len(params)
        l2norms = [torch.norm(update).item() for update in params]
        M = np.median(l2norms)
        L = 0.1
        R = 3.0
        S1_idxs = []
        for idx, (l2norm, update) in enumerate(zip(l2norms, params)):
            if l2norm >= L * M and l2norm <= R * M:
                S1_idxs.append(idx)

        features = []
        num_para = len(params[0])
        for update in params:
            feature0 = (update > 0).sum().item() / num_para
            feature1 = (update < 0).sum().item() / num_para
            feature2 = (update == 0).sum().item() / num_para

            features.append([feature0, feature1, feature2])

        kmeans = KMeans(n_clusters=2, random_state=0).fit(features)

        flag = 1 if np.sum(kmeans.labels_) > num // 2 else 0
        S2_idxs = list(
            [idx for idx, label in enumerate(kmeans.labels_) if label == flag]
        )

        selected_idx = list(set(S1_idxs) & set(S2_idxs))
        values = params[selected_idx].mean(dim=0)
        return values

    
    def __get_norm_contributions(
        self,
        params: torch.Tensor,
        comb: dict,
    ):
        shapley_values = torch.zeros(len(params))
        norms = torch.norm(params, dim=1)
        norms = (norms - norms.mean()) / (norms.std() + 1e-10)
        for i in range(len(params)):
            for c in comb[i]:
                consistency_without_i = -torch.var(norms[c])
                consistency_with_i = -torch.var(norms[c + [i]])
                shapley_values[i] += consistency_with_i - consistency_without_i
            shapley_values[i] /= len(comb[i])
        return shapley_values


    def __get_direction_contributions(
        self,
        params: torch.Tensor,
        comb: dict,
    ):
        shapley_values = torch.zeros(len(params))
        num = len(params)
        dis_max = torch.zeros((num, num))
        for i in range(num):
            for j in range(i + 1, num):
                dis_max[i, j] = 1 - F.cosine_similarity(
                    params[i, :], params[j, :], dim=0
                )
                dis_max[j, i] = dis_max[i, j]
        for i in range(len(params)):
            for c in comb[i]:
                consistency_without_i = -torch.mean(self.__get_all_cos_sim(dis_max, c))
                consistency_with_i = -torch.mean(self.__get_all_cos_sim(dis_max, c + [i]))
                shapley_values[i] += consistency_with_i - consistency_without_i
            shapley_values[i] /= len(comb[i])
        
        return shapley_values
    
    def __get_all_cos_sim(self, dis_max, idxs):
        c = list(combinations(idxs, 2))
        res = torch.zeros(len(c))
        for i, (x, y) in enumerate(c):
            res[i] = dis_max[x, y]
        return res

    
    def _shapley(
        self,
        params: torch.Tensor,
        *args, **kwargs,
    ):
        num = len(params)
        comb = {}
        for i in range(num):
            remaining_agents = [x for x in range(num) if x != i]
            comb_num_agents_minus_1 = list(combinations(remaining_agents, num - 1))
            comb_num_agents_minus_2 = list(combinations(remaining_agents, num - 2))
            comb[i] = [list(c) for c in (comb_num_agents_minus_1 + comb_num_agents_minus_2)]
        if all(c == 0 for c in kwargs["contributions"]):
            task_selected = list(range(num))
        else:
            task_contributions = torch.tensor(kwargs["contributions"], dtype=torch.float)
            task_selected = torch.argsort(task_contributions, descending=True).tolist()[:-1]
        norm_contributions = self.__get_norm_contributions(params, comb)
        norm_selected = torch.argsort(norm_contributions, descending=True).tolist()[:-1]
        direction_contributions = self.__get_direction_contributions(params, comb)
        direction_selected = torch.argsort(direction_contributions, descending=True).tolist()[:-1]
        selected_idx = list(set(task_selected) & set(norm_selected) & set(direction_selected))
        values = params[selected_idx].mean(dim=0)
        return values