# Scenario A baseline fidelity contract

The three baselines are **paper-adapted implementations**, not bit-for-bit
reproductions of the authors' code. They share Scenario A's CIFAR-100 split,
client/channel trace, available MIGs, total spectrum, USFL model, number of
communication rounds, optimizer budget and evaluation schedule. Each method keeps
its own paper-defined controller objective; the comparison report uses the same
external delay, reward and accuracy metrics for all methods.

## CPSL

Retained mechanisms:

- one long-timescale global split pair selected by sample-average approximation
  on the excluded warm-up observations;
- capacity-constrained Gibbs client clustering;
- greedy marginal-delay subchannel allocation.

Scenario A adaptation: the paper executes clusters sequentially and reuses the full
spectrum in every cluster. Scenario A exposes concurrent MIGs, so subchannels are
allocated once from the same global spectrum budget used by every compared method.

## ClusterSFL

Retained mechanisms:

- fixed model partition;
- label-distribution (symmetric KL) worker grouping;
- top-worker selection by ingress channel;
- completion-time-aligned feature compression;
- cluster local-update-frequency and data/frequency aggregation weights.

Scenario A adaptation: the paper's top worker is represented by the edge-cluster
coordinator because Scenario A executes the middle model on a MIG. A deterministic
top-k operator instantiates the selected feature-compression ratio on the real
smashed tensor before the middle model.

## PCSFL

Retained mechanisms:

- model-parameter PCA summary, client data volume, channel and compute state;
- recurrent state encoder and independent clustering/splitting Q heads;
- independent online/target Double-DQN targets;
- nonlinear Wasserstein-model-distance and waiting-factor reward;
- per-cluster training, data-weighted edge aggregation and data-weighted cloud
  aggregation;
- equal bandwidth (PCSFL has no bandwidth action).

Scenario A adaptation: Scenario A has one station model before the first round rather
than a persistent model per active client. The first state therefore uses its bounded
PCA summary; later states retain each client's most recent cluster-model summary.
After the action, separate cluster models are trained and aggregated at the
edge/cloud boundaries available in Scenario A.

## Claim boundary

A result may be described as a comparison with the three paper-adapted baselines.
It must not be described as reproducing the numerical tables of the source papers,
because their datasets, networks, wireless systems and execution hardware differ.
