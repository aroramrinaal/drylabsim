window.DRYLABSIM_LAB_COMMANDS = {
  run_qc:                  ['$ scanpy.pp.filter_cells()', '  filtering 11847 cells...', '  10234 passed QC', '  doublet rate: 3.2%'],
  normalize_data:          ['$ scran.normalize(adata)', '  computing size factors...', '  log1p transform', '  HVGs: 3000 selected'],
  cluster_cells:           ['$ sc.tl.leiden(adata, 0.8)', '  building kNN graph...', '  optimizing modularity', '  14 clusters found'],
  differential_expression: ['$ DESeq2.run(IPF, Ctrl)', '  fitting GLM...', '  1847 DE genes', '  SPP1 log2FC=3.42 ***'],
  pathway_enrichment:      ['$ gseapy.enrich(de_genes)', '  KEGG + Reactome...', '  ECM-receptor p=4.2e-12', '  TGF-beta p=1.8e-09'],
  marker_selection:        ['$ rank_markers(candidates)', '  SPP1  AUROC: 0.94', '  MMP7  AUROC: 0.87', '  COL1A1 AUROC: 0.81'],
  validate_marker:         ['$ cross_validate("SPP1")', '  fold 1: 0.93', '  fold 2: 0.89', '  mean AUROC: 0.91 OK'],
};
