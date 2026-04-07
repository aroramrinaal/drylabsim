window.DRYLABSIM_LAB_COMMANDS = {
  fallback: ['$ processing...', '  computing...', '  done'],
  run_qc:                  ['$ scanpy.pp.calculate_qc_metrics()', '  16204 cells profiled...', '  14583 passed QC', '  ambient RNA elevated post-treatment'],
  filter_data:             ['$ adata = adata[qc_mask]', '  retaining 13912 cells', '  minor relapse cluster preserved', '  aggressive filtering avoided'],
  normalize_data:          ['$ scran.normalize(adata)', '  computing size factors...', '  log1p transform', '  HVGs: 3500 selected'],
  integrate_batches:       ['$ harmony.integrate(pre, relapse)', '  aligning treatment batches...', '  residual batch effect: 0.08', '  relapse-specific blast states retained'],
  cluster_cells:           ['$ sc.tl.leiden(adata, 0.6)', '  building kNN graph...', '  9 clusters found', '  2 relapse-enriched blast clusters split'],
  differential_expression: ['$ rank_genes_groups(relapse, pre)', '  bulk effect is mixed...', '  MCL1 modestly up, JAK2/STAT5 partial', '  no single driver sufficient'],
  pathway_enrichment:      ['$ gseapy.enrich(cluster_specific_hits)', '  clone A: apoptosis escape', '  clone B: JAK-STAT survival', '  branch-specific programs recovered'],
  regulatory_network_inference: ['$ scenic.run(adata)', '  STAT5 regulon active in clone B', '  CREB/ATF anti-apoptotic module in clone A', '  founder program persists across branches'],
  trajectory_analysis:     ['$ paga + monocle3', '  founder blast state identified', '  2 resistant branches after treatment', '  pseudotime supports parallel evolution'],
  marker_selection:        ['$ rank_markers(resistant_clusters)', '  clone A: MCL1 BCL2A1 SOX4', '  clone B: JAK2 PIM1 SOCS2', '  clone-resolved markers saved'],
};
