Batchers
========

Batchers are objects allowing the execution of multiple policies over multiple environments, using multiple processes. We provide two batchers:
1) `EpisodeBatcher` to acquire complete episodes (until a `done` is reached)
2) `Batcher` to acquire the `n` next timesteps over multiple environments

Examples about how to use batchers are given in the other sections.

**Important**: All the information (e.g `DictTensor`) produced by both the `Agent` and the `Env` will be available to the user in the returned `TemporalDictTensor`

Parallelization Schema
----------------------

The generic parallelization schema is illustated in the following picture.

.. image:: http://www-connex.lip6.fr/~denoyer/wordpress/wp-content/uploads/2014/09/41416063_1843670485712946_6632995093617836032_n.jpg
  :width: 1024
  :alt: Parallelization Schema
