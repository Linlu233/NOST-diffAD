# Formula Trace

This file maps formulas in `NOST-DiffAD.md` to the implementation.

| Document Item | Formula | Implementation |
|---|---|---|
| Patch tokens | `X={x_i}`, `z_i=f_vfm(x_i)` | `FeatureExtractor` in `src/nostdiffad/features.py` |
| Graph adjacency | `A_ij=sigma(beta_s cos(z_i,z_j)+beta_p exp(-||p_i-p_j||^2/sigma_p^2)+beta_m I[m_i=m_j])` | `build_patch_graph` in `src/nostdiffad/graph.py` |
| Laplacian | `L_A=D_A-A` | `graph_laplacian` in `src/nostdiffad/graph.py` |
| Attribute branch | `h_i^a=MLP_a([z_i,W_wave(x_i),p_i])` | `AttributeBranch` in `src/nostdiffad/model.py` |
| Structure branch | `h_i^s=GNN_theta(z_i,A)` | `StructureBranch` in `src/nostdiffad/model.py` |
| Fusion | `LN(h_i^a+h_i^s+Gate([h_i^a,h_i^s])*(h_i^a-h_i^s))` | `FusionBlock` in `src/nostdiffad/model.py` |
| Forward diffusion | `H_t=sqrt(alpha_bar_t)H_0+sqrt(1-alpha_bar_t)epsilon` | `DiffusionSchedule.q_sample` in `src/nostdiffad/diffusion.py` |
| Score matching | `||s_theta(H_t,A,t,c)+epsilon/sqrt(1-alpha_bar_t)||_2^2` | `score_matching_loss` in `src/nostdiffad/losses.py` |
| Prototypes | `min_k ||h_i-p_c,k||_2^2` | `PrototypeBank` in `src/nostdiffad/losses.py` |
| NMF | `H_c^+ approx U_cV_c^T`, `||H_c^+-U_cV_c^T||_F^2+lambda_u||U_c||_1+lambda_v||V_c||_1` | `NMFConstraint.normal_structure_matrix` and `NMFConstraint` in `src/nostdiffad/losses.py` |
| Laplacian smoothness | `Tr(H^T L_A H)` | `laplacian_loss` in `src/nostdiffad/losses.py` |
| Score energy | `int ||s_theta(h_i,t,A,c)||_2^2 dt` | `score_energy_integral` and `EnergyComputer.score_energy` in `src/nostdiffad/energy.py` |
| Topology energy | `sum_j A_ij ||r_i-r_j||_1` | `EnergyComputer.topology_energy` in `src/nostdiffad/energy.py` |
| Wave energy | `||W(x_i)-W(xhat_i)||_1` | `patch_decoder`, `WaveletHighFrequency.patch_features`, and `EnergyComputer.wave_energy` in `src/nostdiffad/model.py`, `src/nostdiffad/wavelet.py`, and `src/nostdiffad/energy.py` |
| Total energy | `alpha E_score+beta E_proto+gamma E_topo+eta E_wave` | `EnergyComputer.total_energy` in `src/nostdiffad/energy.py` |
| Pixel map | `Upsample({E_i})` | `patch_energy_to_pixel_map` in `src/nostdiffad/energy.py` |
| Image score | `TopKMean({E_i},k)` | `topk_mean_score` in `src/nostdiffad/energy.py` |
| Conformal threshold | `Quantile_{1-alpha}(E_cal)` | `ConformalThreshold` in `src/nostdiffad/conformal.py` |
| Robustness tests | `lighting/noise/compression robustness` | `apply_robustness` and `scripts/run_experiment_matrix.py` |
| Cross-category generalization | multi-category training/evaluation | `scripts/run_experiment_matrix.py --cross-category` with `data.category=null` |

Inputs required by the document:

- The `beta_m I[m_i=m_j]` graph term requires SAM/SAM2 part mask ids. Ground-truth anomaly masks are never used as `m_i`.
- The implementation supports Hugging Face DINOv2/CLIP and the official SAM2 image encoder when `model.sam2_config` and `model.sam2_checkpoint` are provided.
