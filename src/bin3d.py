from __future__ import annotations
import numpy as np
from numba import jit
import numba as nb
import logging
from functools import partial, reduce
from itertools import combinations, permutations
import operator
from .item import Item
import networkx as nx
import copy
from scipy.signal import fftconvolve
from typing import List
import time

log = logging.getLogger(__name__)


class Bin():
    """This class represents a three-dimensional bin that can be filled with Items."""

    OBS_META_SIZE = 2

    def __init__(self, extents: list = None, maxweight: int = 100, name: str = None):
        """Creates a new bin."""
        assert all([i > 0 for i in extents])
        assert maxweight >= 0
        self.extents = np.array(extents)
        self.maxweight = maxweight
        self.name = name
        self.items = []
        self.heightmap = np.zeros(self.extents[:2], dtype=int)
        self.items_fit_cache = ItemsFitCache()

    def is_valid(self, item: Item) -> bool:
        """Returns if the given item and its position is valid in this bin."""
        return self.weight_left() >= item.weight and self.is_in_bounds(item) and self.is_flush(item)

    def is_in_bounds(self, item: Item) -> bool:
        """Returns if the given item is in this bin's bounds."""
        return (item.max() <= self.extents).all() & \
               (item.min() >= 0).all()

    def is_flush(self, item: Item) -> bool:
        """Returns if the surface the given item would occupy is flush."""
        ground = self.heightmap[item.min(0):item.max(0), item.min(1):item.max(1)]
        return (item.position[2] == ground).all()

    def min(self, dim) -> int:
        """Returns the minimal coordinate of this bin in the given dimension."""
        return 0

    def max(self, dim) -> int:
        """Returns the maximal coordinate of this bin in the given dimension."""
        return self.extents[dim]

    def add(self, item: Item, position: list) -> bool:
        """Adds the given item to this bin at the given position if possible.
        Returns if adding was successfull."""
        item.set_position(position)
        if self.is_valid(item):
            # add item and update heightmap
            self.items.append(item)
            self.heightmap[item.min(0):item.max(0), item.min(1):item.max(1)] = item.max(2)
            assert self.heightmap.sum() == self.item_volume()
            self.items_fit_cache.clear()
            return True
        else:
            item.clear_position()
            return False

    def undo_add(self) -> Item:
        """Removes the latest added item from this bin."""
        item = self.items.pop()
        self.heightmap[item.min(0):item.max(0), item.min(1):item.max(1)] = item.min(2)
        assert self.heightmap.sum() == self.item_volume()
        item.clear_position()
        self.items_fit_cache.clear()
        return item

    def observation(self) -> np.ndarray:
        """Returns the observation representation of this bin as heightmap and meta info."""
        return self.heightmap, np.array([self.weight_left(), self.extents[2]])

    def __str__(self):
        return (f'Bin: {self.name}; extents: {self.extents}; '
                f'maxweight: {self.maxweight}; items: {len(self.items)}')

    def __repr__(self):
        return str(self)

    def get_2d_meshgrid(self) -> np.ndarray:
        """Returns 2D meshgrid coordinates of this bin."""
        x = np.arange(self.extents[0])
        y = np.arange(self.extents[1])
        return np.meshgrid(x, y, indexing='ij')

    def valid_map(self, item: Item) -> np.ndarray:
        """Returns a map with the valid positions for the given item."""
        hm = self.heightmap
        xx, yy = self.get_2d_meshgrid()
        coords = np.stack([xx, yy], axis=2).astype(np.int)
        xy_bounds = (hm.shape - item.extents[:2])
        is_in_bounds = (coords <= xy_bounds).all(2)
        is_within_z = (self.heightmap + item.extents[2]) <= self.extents[2]
        is_flat = is_flat_jit(hm, item.extents[0], item.extents[1])
        return (is_flat & is_in_bounds & is_within_z)

    def valid_map_fail_safe(self, item: Item, group: list) -> np.ndarray:
        """Returns a map with the valid positions for the given item considering the given group items."""
        valid_map = self.valid_map(item)
        mask = np.zeros_like(valid_map)
        for idx in np.argwhere(valid_map):
            idx3d = np.array([*idx, self.heightmap[tuple(idx)]])
            assert self.add(item, idx3d)
            mask[tuple(idx)] = self.can_fit(group)
            self.undo_add()
        return mask

    def can_fit_any_group(self, groups: List[List[Item]]) -> bool:
        """Returns if this given bin can fit at least one of the given groups."""
        # sort by small extents, small groups
        groups = sorted(groups, key=lambda x: ((x[0].extents**2).sum()+len(x)**2))
        for group in groups:
            if self.can_fit(group):
                return True
        return False

    def get_closest_valid(self, x: int, y: int, item: Item) -> tuple:
        """Returns a valid position which is closest to the given location."""
        valid = self.valid_map(item)
        if not valid.any():
            return None
        xx, yy = self.get_2d_meshgrid()
        distance = np.abs(xx-x) + np.abs(yy-y)
        assert valid.shape == distance.shape, \
               f'{valid.shape} and {distance.shape}'
        idxvalid = np.argwhere(valid)
        closest_idx = distance[idxvalid[:, 0], idxvalid[:, 1]].argmin()
        return tuple(idxvalid[closest_idx])

    def weight_left(self) -> int:
        """Returns the weight left in this bin."""
        return self.maxweight - self.items_weight()

    def items_weight(self) -> int:
        """Returns the weight of the items in this bin."""
        return sum([i.weight for i in self.items])

    def volume(self) -> int:
        """Returns this bins' volume."""
        return reduce(operator.mul, self.extents)

    def item_volume(self) -> int:
        """Returns this bins' items' volume."""
        return np.sum(list(map(Item.volume, self.items)))

    def volume_percentage(self) -> float:
        """Returns the percent of volume occupied in this bin."""
        return self.item_volume() / self.volume()

    def volume_left(self) -> int:
        """Returns the amount of volume left in this bin."""
        return self.volume() - self.item_volume()

    def height_variance(self) -> float:
        """Returns the variance of heights in this bin."""
        return self.heightmap.var()

    def height_max(self) -> int:
        """Returns the maximal height reached by an item in this bin."""
        return self.heightmap.max()

    def height_percentage(self) -> float:
        """Returns the percentage of height reached by the items in this
        bin."""
        return self.heightmap.max() / self.extents[2]

    def diff_percentage(self) -> float:
        """Returns the percentage of nonzero diffs in this bin."""
        diff_x = np.diff(self.heightmap, axis=0) != 0
        diff_y = np.diff(self.heightmap, axis=1) != 0
        size = (diff_x.size + diff_y.size)
        if size == 0:
            size = 1
        return (diff_x.sum() + diff_y.sum()) / size

    def bounding_box(self) -> np.array:
        """Returns the bounding box of the items in this bin based on the heightmap."""
        nonzero = self.heightmap != 0
        x_nonzero = np.arange(self.heightmap.shape[0])[nonzero.any(0)]
        y_nonzero = np.arange(self.heightmap.shape[1])[nonzero.any(1)]
        x0 = x_nonzero.min()
        x1 = x_nonzero.max()
        y0 = y_nonzero.min()
        y1 = y_nonzero.max()
        z1 = self.heightmap.max()
        return np.array([x1-x0+1, y1-y0+1, z1])

    def copy(self) -> Bin:
        """Returns a deepcopy of this Bin."""
        return copy.deepcopy(self)

    def _can_fit_pre_check(self, items: list):
        assert len(items) > 0
        # volume check
        if self.volume_left() < sum([i.volume() for i in items]):
            return False
        # extents check
        assert all([items[0].equals_physical(i) for i in items]), 'expecting only physically equal items'
        if (np.sort(self.extents) < np.sort(items[0].extents)).any():
            return False
        # weight check
        if self.weight_left() < sum([i.weight for i in items]):
            return False
        return True

    def can_fit(self, items: list):
        """Returns if the given items can fit in this bin. This function uses a cache."""
        items = [items] if isinstance(items, Item) else items
        if len(items) == 0:
            return True
        if not self._can_fit_pre_check(items):
            return False
        # cache check
        item = items[0]
        key = np.concatenate([np.sort(item.extents), [item.weight], [len(items)]])
        cache_hit = self.items_fit_cache.get(key)
        if cache_hit is not None:
            log.debug('can fit items cache hit')
            return cache_hit.any()
        # calculation
        log.debug('can fit items cache miss')
        can_fit = _can_fit_recursive_jit(
            self.heightmap.copy(),
            np.vstack([i.extents for i in items]),
            np.repeat(Item.N_ROTATIONS, len(items)),  # rotations
            np.vstack(list(permutations([0, 1, 2]))),
            self.extents[2]
        )
        self.items_fit_cache.add(key, can_fit)
        return can_fit

    def can_fit_locked_rotation(self, item_locked_rotation: Item, items: list = []):
        """Checks if the items can fixed with one locked rotation. Uses no cache."""
        assert isinstance(items, list)
        items = items + [item_locked_rotation]
        if not self._can_fit_pre_check(items):
            return False
        rotations = np.repeat(6, len(items))
        # fixes the rotation of one item that is placed first
        rotations[-1] = 1  # jit check loop later only checks one rotation for locked item (identity)
        can_fit = _can_fit_recursive_jit(
            self.heightmap.copy(),
            np.vstack([i.extents for i in items]),
            rotations,
            np.vstack(list(permutations([0, 1, 2]))),
            self.extents[2]
        )
        return can_fit

    def __eq__(self, other: Bin) -> bool:
        return (
            (self.extents == other.extents).all() &
            (self.maxweight == other.maxweight) &
            (self.name == other.name) &
            (self.items == other.items) &
            (self.heightmap == other.heightmap).all()
        )

    def __hash__(self):
        return hash((
            tuple(self.extents),
            self.maxweight,
            self.name,
            tuple(self.items),
            # tuple(self.heightmap.flatten()),
        ))
    
    def physical_hash(self):

        item_hashes = [item.physical_hash() for item in self.items]
        return hash((
            tuple(self.extents),
            tuple(item_hashes)))    


@nb.njit
def add_item(x, i, j, item):
    x[i:i+item[0], j:j+item[1]] = x[i:i+item[0], j:j+item[1]] + item[2]


@nb.njit
def remove_item(x, i, j, item):
    x[i:i+item[0], j:j+item[1]] = x[i:i+item[0], j:j+item[1]] - item[2]


@nb.njit
def _can_fit_recursive_jit(x, items, rotations, permutations, maxz):
    if len(items) == 0:
        return True
    item_orig = items[-1]
    for rotation in np.arange(rotations[-1]):
        item = np.take(item_orig, permutations[rotation])
        ex, ey, ez = item[0], item[1], item[2]
        for i in np.arange(x.shape[0]+1-ex):
            for j in np.arange(x.shape[1]+1-ey):
                if _local_is_flat(x[i: i+ex, j: j+ey]) and x[i, j] + ez <= maxz:
                    add_item(x, i, j, item)
                    if _can_fit_recursive_jit(x, items[:-1], rotations[:-1], permutations, maxz):
                        return True
                    remove_item(x, i, j, item)
    return False


@jit(nopython=True)
def is_flat_map(xy: list, lwh: list, hm: np.ndarray) -> bool:
    """Returns if the given given heightmap (hm) is flat at the given
    coordinates (xy) for the given length, width and height (lwh)."""
    ground = hm[xy[0]:xy[0]+lwh[0], xy[1]:xy[1]+lwh[1]]  # ground of item
    return (ground[0, 0] == ground).all()  # all ground heights are equal


def _get_force(force_graph: nx.DiGraph, node: str) -> int:
    """Function to recursivly calculate force of the given node."""
    children = [edge[1] for edge in force_graph.out_edges(node)]
    force_incoming = 0
    pressure_incoming = 0
    for edge in force_graph.out_edges(node):
        # force
        area_ratio = force_graph.edges[edge]['support_area_ratio']
        force = _get_force(force_graph, edge[1]) * area_ratio
        force_incoming += force
        force_graph.edges[edge]['force'] = force
        # pressure
        area = force_graph.edges[edge]['support_area']
        pressure = force / area
        pressure_incoming += pressure
        force_graph.edges[edge]['pressure'] = pressure
    force_sum = force_incoming + force_graph.nodes[node]['weight']
    pressure_sum = pressure_incoming + force_graph.nodes[node]['weight']
    force_graph.nodes[node]['force'] = force_incoming
    force_graph.nodes[node]['pressure'] = pressure_incoming
    return force_sum


def get_force_graph(items: list) -> nx.DiGraph:
    """Sets up a force graph for the given items."""
    force_graph = nx.DiGraph()
    # add floor, area is not relevant so set it to -1
    force_graph.add_node('floor', subset=-1, weight=0, force=0, area=-1, idx=-1)
    # add all items as nodes with their respective weight, area and initial force sum of 0
    # the subset setting is later used by the multipartite_layout to order the items by their z position
    for i, item in enumerate(items):
        force_graph.add_node(item.id, subset=item.min(2), weight=item.weight, force=0, area=item.xy_area(), idx=i)
        if item.min(2) == 0:  # item is located on the floor (full support ratio of 1.0)
            force_graph.add_edge('floor', item.id, support_area=item.xy_area(), support_area_ratio=1.0)
    # create edges for each item support
    for a, b in combinations(items, 2):
        support_area = a.support_area(b)
        if support_area > 0:  # if a supports b
            # the ratio is later used to distribute force, the area is later used to calculate pressure
            support_area_ratio = support_area / b.xy_area()
            force_graph.add_edge(a.id, b.id, support_area=support_area, support_area_ratio=support_area_ratio)
    _get_force(force_graph, 'floor')
    return force_graph


@nb.njit
def _local_is_flat(x: np.ndarray) -> bool:
    ref = x[0, 0]
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] != ref:
                return False
    return True


@nb.njit()
def is_flat_jit(x: np.ndarray, ex: int, ey: int) -> np.ndarray:
    out = np.zeros_like(x)
    for i in np.arange(x.shape[0]+1-ex):
        for j in np.arange(x.shape[1]+1-ey):
            out[i, j] = _local_is_flat(x[i: i+ex, j: j+ey])
    return out


@nb.njit
def can_fit_lazy_jit(x: np.ndarray, ex: int, ey: int, ez: int, zmax: int) -> bool:
    for i in range(x.shape[0]+1-ex):
        for j in range(x.shape[1]+1-ey):
            if _local_is_flat(x[i: i+ex, j: j+ey]) and x[i, j] + ez <= zmax:
                return True
    return False


def is_flat_meancompare(x: np.ndarray, ex: int, ey: int) -> np.ndarray:
    """Calculates which areas are flat by checking if geometric mean and arithmetic mean are equal."""
    x = x + 1  # prevent division by 0
    mean_kernel = np.full((ex, ey), 1/(ex*ey))
    x_mean = fftconvolve(x, mean_kernel, mode='valid')[:-1, :-1]
    x_logmean = fftconvolve(np.log(x), mean_kernel, mode='valid')[:-1, :-1]
    # compare in np.log space, log is cheaper than np.exp and slice
    is_flat = np.isclose(np.exp(x_logmean), x_mean, rtol=1e-6)
    return np.pad(is_flat, ((0, ex), (0, ey)))


class ItemsFitCache:
    """Cache used to store if a set of same Items are able to fit in the current bin."""
    def __init__(self):
        self.cache = None
        self.values = None
        self.counter = {'hit': 0, 'miss': 0}

    def get(self, key: np.array, return_idxs: bool = False) -> np.array:
        """Returns the cached value for the given key. A hit is defined as same extents
        and the key having leq weight and count than the cache value."""
        if self.cache is None:
            self.counter['miss'] = self.counter['miss'] + 1
            return None
        # first stage, extents
        extents_hits = np.isclose(key[:3], self.cache[:, :3]).all(1)  # compare extents
        extent_hit_idxs = np.argwhere(extents_hits).squeeze(1)
        if len(extent_hit_idxs) == 0:
            self.counter['miss'] = self.counter['miss'] + 1
            return None
        # second stage, weight and count
        # if weight and count are leq the searched key, their values can be used
        # this means, less can be packed, if it is more weight or count or both, computation is needed
        full_hits = (self.cache[extent_hit_idxs][:, 3:] <= key[3:]).all(1)
        full_hit_idx = np.argwhere(full_hits).squeeze(1)
        if len(full_hit_idx) == 0:
            self.counter['miss'] = self.counter['miss'] + 1
            return None
        self.counter['hit'] = self.counter['hit'] + 1
        idxs = extent_hit_idxs[full_hit_idx]
        values = self.values[idxs]
        assert all([values[0] == i for i in values]), key  # all cache hits should have the same value
        value = values[0]
        if return_idxs:
            return value, idxs
        return value

    def add(self, key: np.array, value: bool):
        """Adds the given key value pair to this cache."""
        if self.cache is not None:
            self.cache = np.vstack([self.cache, key])
            self.values = np.append(self.values, value)
        else:
            self.cache = key.reshape(1, -1)
            self.values = np.array([value])

    def size(self) -> int:
        """Returns this cache's size."""
        return len(self.cache) if self.cache is not None else 0

    def clear(self):
        """Clears this cache."""
        self.cache = None
        self.values = None