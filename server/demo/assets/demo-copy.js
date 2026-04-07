window.DRYLABSIM_DEMO_COPY = {
  status: {
    ready: 'Ready',
    running: 'Running',
    complete: 'Complete',
  },
  buttons: {
    run: 'Run Episode',
    running: 'Running...',
  },
  labels: {
    stepRewardDefault: '--',
  },
  introLines: [
    '<span class="t-label">[DryLabSim]</span> <span class="t-dim">Initializing environment...</span>',
    '<span class="t-label">[DryLabSim]</span> Scenario: <span class="t-str">biomarker_validation_lung</span> (Hard)',
    '<span class="t-label">[DryLabSim]</span> Organism: <span class="t-str">Homo sapiens</span> | Tissue: <span class="t-str">Lung</span>',
    '<span class="t-label">[DryLabSim]</span> Budget: <span class="t-num">$100,000</span> | Time: <span class="t-num">180 days</span> | Max steps: <span class="t-num">30</span>',
    '<span class="t-label">[DryLabSim]</span> Task: Validate <span class="t-kw">SPP1</span> as biomarker for idiopathic pulmonary fibrosis',
  ],
  completionLines: [
    '<span class="t-label">[DryLabSim]</span> <span class="t-ok">Episode complete!</span>',
    '<span class="t-label">[DryLabSim]</span> Total reward: <span class="t-ok">+{{cumReward}}</span> | Steps: <span class="t-num">{{stepCount}}</span> | Budget remaining: <span class="t-num">$65,000</span>',
    '<span class="t-label">[DryLabSim]</span> Literature match: <span class="t-ok">4/5 expected findings confirmed</span>',
    '<span class="t-label">[DryLabSim]</span> Calibration: <span class="t-ok">Well-calibrated</span> (no overconfidence penalty)',
  ],
  resetLine: '<span class="t-dim">Environment reset. Click "Run Episode" to start.</span>',
  initialLines: [
    '<span class="t-dim">DryLabSim v1.0 | biomarker_validation_lung</span>',
    '<span class="t-dim">Click "Run Episode" to start the demo.</span>',
  ],
};
