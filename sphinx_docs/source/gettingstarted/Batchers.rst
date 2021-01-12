Batchers
========

Batchers are object allowing to execute multiple policies over multiple environments, using multiple processes. We provide two batchers:
1) `EpisodeBatcher` to acquire complete episodes (until a `done` is reached)
2) `Batcher` to acquire the `n` next timesteps over multiple environments

Examples about how to use batchers are given in the other sections.

**Important**: All the information (e.g `DictTensor`) produced by both the `Agent` and the `Env` will be available to the user.
