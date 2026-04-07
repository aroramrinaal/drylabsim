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
    '<span class="t-label">[DryLabSim]</span> Scenario: <span class="t-str">venetoclax_resistance_multiclone</span> (Expert)',
    '<span class="t-label">[DryLabSim]</span> Organism: <span class="t-str">Homo sapiens</span> | Tissue: <span class="t-str">Bone marrow</span>',
    '<span class="t-label">[DryLabSim]</span> Budget: <span class="t-num">$150,000</span> | Time: <span class="t-num">210 days</span> | Max steps: <span class="t-num">30</span>',
    '<span class="t-label">[DryLabSim]</span> Task: Resolve parallel <span class="t-kw">venetoclax resistance</span> mechanisms across AML subclones',
  ],
  completionLines: [
    '<span class="t-label">[DryLabSim]</span> <span class="t-ok">Episode complete!</span>',
    '<span class="t-label">[DryLabSim]</span> Total reward: <span class="t-ok">+{{cumReward}}</span> | Steps: <span class="t-num">{{stepCount}}</span> | Budget remaining: <span class="t-num">$117,000</span>',
    '<span class="t-label">[DryLabSim]</span> Mechanism match: <span class="t-ok">2 resistant programs recovered across 2 relapse clones</span>',
    '<span class="t-label">[DryLabSim]</span> Calibration: <span class="t-ok">Cautious dual-clone synthesis</span> with lower confidence on the minor branch',
  ],
  resetLine: '<span class="t-dim">Environment reset. Click "Run Episode" to start.</span>',
  initialLines: [
    '<span class="t-dim">DryLabSim v1.0 | venetoclax_resistance_multiclone</span>',
    '<span class="t-dim">Click "Run Episode" to start the demo.</span>',
  ],
};
