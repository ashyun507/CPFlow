# Cyclic Peptide Model Components

This directory contains the non-training, non-inference source code used to describe the model architecture and cyclic peptide encoding components.

The release is intended as a lightweight code reference for an anonymous repository. It does not include masked flow matching training code, sampling code, test-time scaling code, checkpoints, datasets, or evaluation outputs.

## Contents

```text
models_con/
  node.py                 Residue/node feature encoder
  edge.py                 Pair/edge feature encoder with cyclic peptide priors
  cyclic_edge_priors.py   CPCore-derived cyclic C-beta distance prior loader
  ga.py                   Geometry-aware IPA/Transformer trunk
  ipa_pytorch.py          Invariant point attention and structure update blocks
  torsion.py              Torsion and side-chain reconstruction utilities
  utils.py                Shared positional embedding and tensor helpers

pepflow/modules/
  common/geometry.py      Rigid-body and geometry utilities
  common/layers.py        Angular/distance encoding layers
  common/topology.py      Chain topology helpers
  protein/constants.py    Protein atom and residue constants

openfold/utils/
  rigid_utils.py          Rigid transformation utilities used by IPA blocks
```

## Model Components

The architecture is organized around residue-level node features, residue-pair edge features, and a geometry-aware trunk.

`NodeEmbedder` encodes amino-acid identity, chain-local residue position, local backbone coordinates, and backbone torsion features into residue embeddings.

`EdgeEmbedder` encodes amino-acid pair identity, chain-local relative position, atom-pair distances, and pairwise orientation features into pair embeddings.

`GAEncoder` applies invariant point attention, sequence transformer layers, node transitions, edge transitions, and rigid-frame updates to jointly update sequence and structure representations.

## Cyclic Peptide Encoding

The cyclic peptide prior is implemented in `models_con/cyclic_edge_priors.py` and consumed by `models_con/edge.py`.

The code supports three cyclization classes:

```text
HEADTAIL
ISOPEPTIDE
DISULFIDE
```

For cyclic peptide mode, the edge encoder adds a 128-bin C-beta distance prior to selected peptide edges:

```text
adjacent peptide edges: i <-> i+1
terminal peptide edge: first <-> last
```

The prior vectors are type-specific. They are selected by `cyclic_type_id` and concatenated into the pair feature before the final edge MLP.

## Data Dependency For Cyclic Priors

`cyclic_edge_priors.py` expects the CPCore-derived prior files at:

```text
datasets/cpcore/CPCore_properties/CPCore_Cb_distance_binned_128.tsv
datasets/cpcore/CPCore_properties/CPCore_Basic.tsv
datasets/cpcore/CPCore_properties/CPCore_peptide_adjacent_cb_distances.tsv
```

These files are not included here. If only reading the implementation, they are not needed. If importing `models_con.cyclic_edge_priors` directly, provide these files or replace the loader with fixed prior tensors.

## Excluded Code

This lightweight release intentionally excludes:

```text
masked flow matching training objectives
sampling and inference scripts
test-time scaling / SMC search code
checkpoints and generated structures
dataset preprocessing pipelines
```
