# Scenario A baseline fidelity contract

The three baselines are **paper-adapted implementations**, not bit-for-bit
reproductions of the authors' code. They share Scenario A's CIFAR-100 split,
client/channel trace, available MIGs, total spectrum, USFL model, number of
communication rounds, mini-batch/optimizer-step budget and evaluation schedule. Each method keeps
its own paper-defined controller objective; the comparison report uses the same
external delay, reward and accuracy metrics for all methods.

For those shared external metrics, formal Scenario A runs follow research-plan
Section 3.1.1: the uplink rate uses each client's allocated bandwidth share and
Shannon spectral efficiency, while downlink rates are allocated centrally by the
base station and are not agent actions. Communication bytes are measured from the
actual batched boundary tensors accumulated over the local optimizer steps. Under
the stated forward/backward symmetry, each direction carries
`v(l1) + v(l2)`: the uplink carries the `l1` activation and `l2` gradient, and the
downlink carries the `l2` activation and `l1` gradient. The default Scenario A base-
station scheduler gives clients equal downlink bandwidth; callers may instead pass
explicit downlink rates without changing any baseline controller.

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
- the paper local-update-frequency signal as a reported diagnostic.

Scenario A adaptation: the paper's top worker is represented by the edge-cluster
coordinator because Scenario A executes the middle model on a MIG. A deterministic
top-k operator instantiates the selected feature-compression ratio on the real
smashed tensor before the middle model. The primary fixed-150-round comparison
executes the shared `local_steps` budget for every active client and aggregates by
the mini-batch samples actually consumed. This prevents communication-round parity
from hiding a several-fold optimizer-step mismatch. A paper-faithful variable-
frequency run must be reported separately as accuracy-versus-wall-clock, not mixed
into the fixed-budget final-accuracy comparison.

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
After the action, clients sharing `(cluster, l1, l2)` are executed as one GPU batch;
different split graphs remain separate gradient-accumulation groups. Cluster models are aggregated at the
edge/cloud boundaries available in Scenario A.

## Claim boundary

A result may be described as a comparison with the three paper-adapted baselines.
It must not be described as reproducing the numerical tables of the source papers,
because their datasets, networks, wireless systems and execution hardware differ.
