# Bifrost Roadmap

This is a high-level outline of Bifrost development plans. Unless otherwise
stated, the items on this page have not yet been developed.

## Algorithms and blocks

 * Single-pulse search algorithms
   * Baseline removal, peak finding
 * Pulsar search algorithms
   * Harmonic summing, folding
 * Calibration and imaging algorithms
   * Gridding/degridding, compressive sensing, CLEAN
 * I/O (source/sink) blocks for additional astronomy/audio/generic file formats

## Pipeline features

 * Method of sending data between different servers
 * Remote control mechanisms
 * Pipeline status and performance monitoring
 * Streaming data visualisation

## Backend features

 * Improved packet capture/transmission framework
 * Support for InfiniBand verbs
 * CPU backends for existing CUDA-only algorithms
 * Support for inter-process shared memory rings
 * Optimisations for low-latency applications

## Platform and dependency updates

 * Python 2.x will no longer be supported after the end of 2022.
