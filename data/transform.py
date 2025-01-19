import torch
import numpy as np
from torch_geometric.utils import to_dense_adj, dense_to_sparse


def universal_svd(graphs, threshold=0.2):
    """
    Estimate a graphon by universal singular value thresholding.
    Reference:
    Chatterjee, Sourav.
    "Matrix estimation by universal singular value thresholding."
    The Annals of Statistics 43.1 (2015): 177-214.

    :param graphs: a tensor (batch_size, N, N) or (N, N) adjacency matrices
    :param threshold: the threshold for singular values

    :return: graphon: the estimated (batch_size, N, N) or (N, N) graphon model
    """
    _, num_node, dim_node = graphs.shape

    u, s, vh = torch.linalg.svd(graphs)
    singular_threshold = threshold * (dim_node ** 0.5)
    binary_s = torch.lt(s, singular_threshold)
    s[binary_s] = 0
    graphons = torch.bmm(torch.bmm(u, torch.diag_embed(s)), vh)
    graphons[graphons > 99] = 1
    graphons[graphons < -99] = 0
    return graphons


def create_mask(data, yl, yh, xl, xh):
    dim_freqband_order, dim_node = data.x.shape
    device = data.x.device
    base_edge_index = []
    for row in range(dim_node):
        for col in range(row + 1, dim_node):
            base_edge_index.append([row, col])
    base_edge_index = torch.tensor(base_edge_index, dtype=torch.long).t()

    node_mask = torch.zeros_like(data.x, dtype=torch.bool)
    edge_mask = torch.tensor([], dtype=torch.bool, device=device)
    for offset in range(int(dim_freqband_order / dim_node)):
        node_mask[(yl + offset * dim_node):(yh + offset * dim_node), xl:xh] = True
        node_mask[(xl + offset * dim_node):(xh + offset * dim_node), yl:yh] = True
        edge_mask = torch.cat((edge_mask, node_mask[base_edge_index[0] + offset * dim_node, base_edge_index[1]]), dim=0)

    return node_mask, edge_mask


def batch_shuffle(data):
    data_clone = data.clone()
    batch_size = len(data_clone)

    perm = torch.randperm(batch_size)

    node_offset = int(data_clone.x.shape[0] / batch_size)
    node_perm = perm.repeat_interleave(node_offset) * node_offset + torch.arange(0, node_offset).repeat(batch_size)

    edge_offset = int(data_clone.edge_attr.shape[0] / batch_size)
    edge_perm = perm.repeat_interleave(edge_offset) * edge_offset + torch.arange(0, edge_offset).repeat(batch_size)

    # data_clone = data_clone[perm]
    data_clone.x = data_clone.x[node_perm]
    data_clone.edge_attr = data_clone.edge_attr[edge_perm]
    data_clone.demographic_info = data_clone.demographic_info[perm]
    data_clone.y = data_clone.y[perm]
    data_clone.eid = [data_clone.eid[i] for i in perm]
    # data_clone.freqband_order = data_clone.freqband_order[node_perm]
    return data_clone


def graphon_mixup(data1, data2, lam, threshold=0.2):
    batched_x1 = data1.x.reshape(-1, data1.x.shape[1], data1.x.shape[1])
    batched_x2 = data2.x.reshape(-1, data2.x.shape[1], data2.x.shape[1])

    node_graphons1 = universal_svd(batched_x1, threshold=threshold).reshape(-1, data1.x.shape[1])
    node_graphons2 = universal_svd(batched_x2, threshold=threshold).reshape(-1, data2.x.shape[1])

    batched_edge1 = to_dense_adj(data1.edge_index, data1.freqband_order.squeeze(), data1.edge_attr).squeeze()
    batched_edge2 = to_dense_adj(data2.edge_index, data2.freqband_order.squeeze(), data2.edge_attr).squeeze()

    edge_graphons1 = universal_svd(batched_edge1 + batched_edge1.permute(0, 2, 1), threshold=threshold)
    edge_graphons2 = universal_svd(batched_edge2 + batched_edge2.permute(0, 2, 1), threshold=threshold)

    _, edge_graphons1 = dense_to_sparse(torch.triu(edge_graphons1, 1))
    _, edge_graphons2 = dense_to_sparse(torch.triu(edge_graphons2, 1))

    data1.x = node_graphons1 * lam + node_graphons2 * (1 - lam)
    data1.edge_attr = edge_graphons1.unsqueeze(1) * lam + edge_graphons2.unsqueeze(1) * (1 - lam)
    data1.demographic_info = data1.demographic_info * lam + data2.demographic_info * (1 - lam)
    data1.y = data1.y * lam + data2.y * (1 - lam)


def graphon_cutmix(data1, data2, node_mask, edge_mask, lam, threshold=0.2):
    # batched_x1 = data1.x.reshape(-1, data1.x.shape[1], data1.x.shape[1])
    batched_x2 = data2.x.reshape(-1, data2.x.shape[1], data2.x.shape[1])

    # node_graphons1 = universal_svd(batched_x1, threshold=threshold).reshape(-1, data1.x.shape[1])
    node_graphons2 = universal_svd(batched_x2, threshold=threshold).reshape(-1, data2.x.shape[1])

    # batched_edge1 = to_dense_adj(data1.edge_index, data1.freqband_order.squeeze(), data1.edge_attr).squeeze()
    batched_edge2 = to_dense_adj(data2.edge_index, data2.freqband_order.squeeze(), data2.edge_attr).squeeze()

    # edge_graphons1 = universal_svd(batched_edge1 + batched_edge1.permute(0, 2, 1), threshold=threshold)
    edge_graphons2 = universal_svd(batched_edge2 + batched_edge2.permute(0, 2, 1), threshold=threshold)

    # _, edge_graphons1 = dense_to_sparse(torch.triu(edge_graphons1, 1))
    _, edge_graphons2 = dense_to_sparse(torch.triu(edge_graphons2, 1))

    data1.x[node_mask] = node_graphons2[node_mask]
    data1.edge_attr[edge_mask] = edge_graphons2.unsqueeze(1)[edge_mask]
    # data1.x[node_mask] = data2.x[node_mask]
    # data1.edge_attr[edge_mask] = data2.edge_attr[edge_mask]
    data1.demographic_info = data1.demographic_info * lam + data2.demographic_info * (1 - lam)
    data1.y = data1.y * lam + data2.y * (1 - lam)


def rand_bbox(img_shape, lam, margin=0., count=None):
    """
    Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.

    :param img_shape: tuple, graph shape
    :param lam: float, cutmix lambda value
    :param margin: float, percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
    :param count: int, the number of bbox to generate

    :return: bbox: randomly generated bbox position
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def rand_bbox_minmax(img_shape, minmax, count=None):
    """
    Min-Max CutMix bounding-box
    Inspired by Darknet cutmix impl, generates a random rectangular bbox
    based on min/max percent values applied to each dimension of the input image.

    Typical defaults for minmax are usually in the  .2-.3 for min and .8-.9 range for max.

    :param img_shape: tuple, graph shape
    :param minmax: tuple or list, min and max bbox ratios as percent of graph size
    :param count: int, the number of bbox to generate

    :return: bbox: randomly generated bbox position
    """
    assert len(minmax) == 2
    img_h, img_w = img_shape[-2:]
    cut_h = np.random.randint(int(img_h * minmax[0]), int(img_h * minmax[1]), size=count)
    cut_w = np.random.randint(int(img_w * minmax[0]), int(img_w * minmax[1]), size=count)
    yl = np.random.randint(0, img_h - cut_h, size=count)
    xl = np.random.randint(0, img_w - cut_w, size=count)
    yu = yl + cut_h
    xu = xl + cut_w
    return yl, yu, xl, xu


def cutmix_bbox_and_lam(img_shape, lam, ratio_minmax=None, correct_lam=True, count=None):
    """
    Generate bbox and apply lambda correction.
    """
    if ratio_minmax is not None:
        yl, yu, xl, xu = rand_bbox_minmax(img_shape, ratio_minmax, count=count)
    else:
        yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam or ratio_minmax is not None:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam


class Mixup:
    """
    Mixup/Cutmix that applies different params to each element or whole batch

    :param mixup_alpha: float, mixup alpha value, mixup is active if > 0
    :param cutmix_alpha: float, cutmix alpha value, cutmix is active if > 0
    :param cutmix_minmax: List[float], cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None
    :param prob: float, probability of applying mixup or cutmix per batch or element
    :param switch_prob: float, probability of switching to cutmix instead of mixup when both are active
    :param mode: str, how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
    :param correct_lam: bool, apply lambda correction when cutmix bbox clipped by image borders

    :return: graph: mixup and/or cutmix data
    """
    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                 mode='batch', correct_lam=True):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.mode = mode
        self.correct_lam = correct_lam  # correct lambda based on clipped area for cutmix
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)

    def _params_per_elem(self, batch_size):
        lam = np.ones(batch_size, dtype=np.float32)
        use_cutmix = np.zeros(batch_size, dtype=bool)
        if self.mixup_enabled:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand(batch_size) < self.switch_prob
                lam_mix = np.where(
                    use_cutmix,
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size),
                    np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size))
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
            elif self.cutmix_alpha > 0.:
                use_cutmix = np.ones(batch_size, dtype=bool)
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = np.where(np.random.rand(batch_size) < self.mix_prob, lam_mix.astype(np.float32), lam)
        return lam, use_cutmix

    def _params_per_batch(self):
        lam = 1.
        use_cutmix = False
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
                    np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_elem(self, data):
        batch_size = len(data)
        lam_batch, use_cutmix = self._params_per_elem(batch_size)
        data_orig = data.clone()  # need to keep an unmodified original for mixing source
        for i in range(batch_size):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cutmix[i]:
                    dim_freqband_order, dim_node = data[i].x.shape
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        (dim_node, dim_node), lam, ratio_minmax=self.cutmix_minmax,
                        correct_lam=self.correct_lam
                    )
                    node_mask, edge_mask = create_mask(data[i], yl, yh, xl, xh)
                    graphon_cutmix(data[i], data_orig[j], node_mask, edge_mask, lam)
                else:
                    graphon_mixup(data[i], data_orig[j], lam)

    def _mix_pair(self, data):
        batch_size = len(data)
        lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
        data_orig = data.clone()  # need to keep an unmodified original for mixing source
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cutmix[i]:
                    dim_freqband_order, dim_node = data[i].x.shape
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        (dim_node, dim_node), lam, ratio_minmax=self.cutmix_minmax,
                        correct_lam=self.correct_lam
                    )
                    node_mask, edge_mask = create_mask(data[i], yl, yh, xl, xh)
                    graphon_cutmix(data[i], data_orig[j], node_mask, edge_mask, lam)
                    graphon_cutmix(data[j], data_orig[i], node_mask, edge_mask, lam)
                else:
                    graphon_mixup(data[i], data_orig[j], lam)
                    graphon_mixup(data[j], data_orig[i], lam)

    def _mix_batch(self, data):
        lam, use_cutmix = self._params_per_batch()
        if lam == 1.:
            return 1.

        dim_freqband_order, dim_node = data.x.shape
        shuffled_data = batch_shuffle(data)
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                (dim_node, dim_node), lam, ratio_minmax=self.cutmix_minmax,
                correct_lam=self.correct_lam
            )
            node_mask, edge_mask = create_mask(data, yl, yh, xl, xh)
            graphon_cutmix(data, shuffled_data, node_mask, edge_mask, lam)
        else:
            graphon_mixup(data, shuffled_data, lam)

    def __call__(self, data):
        assert len(data) % 2 == 0, 'Batch size should be even when using this'
        data_clone = data.clone()
        if self.mode == 'elem':
            self._mix_elem(data_clone)
        elif self.mode == 'pair':
            self._mix_pair(data_clone)
        else:
            self._mix_batch(data_clone)
        return data_clone
