#!/usr/bin/env python3
"""Deterministic architecture-grouped K-fold splits.

Drop-in replacement for ``sklearn.model_selection.GroupKFold``.

Why this module exists
----------------------
scikit-learn changed the internal ordering used by ``GroupKFold`` in version
1.6: ``np.argsort(counts)`` became ``np.argsort(counts, kind="stable")``.  Every
architecture in this study contributes either two or three rows, so the count
vector is almost entirely ties and the two orderings produce different folds.
Running the pipeline under scikit-learn >= 1.6 therefore moved cross-validated
numbers (e.g. the CIFAR-10 size-band look-up rule) by up to 1.5 points with no
error and no warning.  Pinning a scikit-learn version would hide the problem
rather than remove it, so the split is computed here instead.

Assignment rule
---------------
Identical to the published analysis: order the groups by descending number of
rows and add each group to the fold that currently holds the fewest rows.  The
order *within* a tied count is the only free parameter, and it is frozen in
``results/cv_folds.json`` -- one entry per (sorted group/count multiset, k)
call site, written when this package was prepared.  Loading the frozen map makes
every script bit-reproducible on any numpy / scikit-learn combination.

If a call site is not in the frozen map (new or filtered data), the map is
extended using numpy's default ``argsort`` order, a note is printed to stderr,
and the new entry is appended to ``results/cv_folds.json`` so the run stays
reproducible from then on.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

FOLD_MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'results', 'cv_folds.json')

_cache: dict | None = None


def _load() -> dict:
    global _cache
    if _cache is None:
        try:
            with open(FOLD_MAP_PATH) as fh:
                _cache = json.load(fh)
        except (OSError, ValueError):
            _cache = {}
    return _cache


def _store(key: str, mapping: dict) -> None:
    cache = _load()
    cache[key] = mapping
    try:
        os.makedirs(os.path.dirname(FOLD_MAP_PATH), exist_ok=True)
        with open(FOLD_MAP_PATH, 'w') as fh:
            json.dump(cache, fh, indent=0, sort_keys=True)
    except OSError:                                    # read-only checkout
        pass


def _key(labels: list[str], counts: list[int], k: int) -> str:
    payload = json.dumps([k, sorted(zip(labels, counts))], separators=(',', ':'))
    return hashlib.sha1(payload.encode()).hexdigest()


def _greedy(labels, counts, k):
    """Descending count, each group to the lightest fold (legacy tie order)."""
    counts = np.asarray(counts)
    order = np.argsort(counts)[::-1]
    load = np.zeros(k)
    fold = {}
    for pos in order:
        lightest = int(np.argmin(load))
        load[lightest] += counts[pos]
        fold[labels[pos]] = lightest
    return fold


def assign(groups, k=5):
    """Return ``{group_label: fold_index}`` for *groups*."""
    g = np.asarray([str(x) for x in np.asarray(groups).ravel()])
    labels, inverse = np.unique(g, return_inverse=True)
    counts = np.bincount(inverse)
    labels = [str(x) for x in labels]
    if k > len(labels):
        raise ValueError(f'n_splits={k} exceeds the number of groups ({len(labels)})')
    key = _key(labels, counts.tolist(), k)
    cached = _load().get(key)
    if cached is not None and set(cached) == set(labels):
        return {lab: int(cached[lab]) for lab in labels}
    fold = _greedy(labels, counts, k)
    print(f'grouped_cv: no frozen split for {len(labels)} groups (k={k}); '
          f'computed one and added it to {os.path.relpath(FOLD_MAP_PATH)}',
          file=sys.stderr)
    _store(key, fold)
    return fold


class GroupKFold:
    """Signature-compatible subset of ``sklearn.model_selection.GroupKFold``."""

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        if shuffle:
            raise ValueError('grouped_cv.GroupKFold does not support shuffle; '
                             'the published analysis never used it')
        self.n_splits = int(n_splits)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        g = np.asarray([str(x) for x in np.asarray(groups).ravel()])
        fold = assign(g, self.n_splits)
        f = np.array([fold[x] for x in g])
        for i in range(self.n_splits):
            test = np.where(f == i)[0]
            train = np.where(f != i)[0]
            yield train, test


__all__ = ['GroupKFold', 'assign', 'FOLD_MAP_PATH']
