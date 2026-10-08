/**
 * Credit Card Fraud Detection ML Engine
 * Simulates Kaggle Credit Card Fraud Dataset (284k schema with V1-V28 PCA features, Time, Amount)
 * Features: Resampling (SMOTE, Undersampling, Class Weighting), Model evaluation, Threshold tuning, SHAP attribution, CSV Batch Import & Prediction
 */

// Random seed generator for reproducible dataset generation
function pseudoRandom(seed) {
  let value = seed;
  return function() {
    value = (value * 9301 + 49297) % 233280;
    return value / 233280;
  };
}

// Generate Box-Muller Gaussian random variable
function randomNormal(mean = 0, std = 1, rng = Math.random) {
  let u1 = rng();
  let u2 = rng();
  while (u1 === 0) u1 = rng();
  const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return z0 * std + mean;
}

// Key Kaggle PCA Features and their typical distribution shifts for Fraud vs Normal
export const FEATURE_PROPERTIES = {
  V14: { name: 'V14 (PCA)', fraudMean: -4.5, normalMean: 0.1, std: 1.2, importance: 0.22, desc: 'Primary PCA Feature (Strongest Fraud Marker)' },
  V12: { name: 'V12 (PCA)', fraudMean: -3.8, normalMean: 0.05, std: 1.1, importance: 0.18, desc: 'Secondary PCA Risk Feature' },
  V10: { name: 'V10 (PCA)', fraudMean: -3.2, normalMean: 0.0, std: 1.0, importance: 0.15, desc: 'Account Velocity Indicator' },
  V4:  { name: 'V4 (PCA)',  fraudMean: 3.6,  normalMean: 0.0, std: 1.1, importance: 0.14, desc: 'Device Anomaly Feature' },
  V11: { name: 'V11 (PCA)', fraudMean: 2.9,  normalMean: 0.0, std: 1.0, importance: 0.11, desc: 'Geolocation Discrepancy' },
  V17: { name: 'V17 (PCA)', fraudMean: -3.1, normalMean: 0.0, std: 1.2, importance: 0.09, desc: 'Behavioral Pattern Shift' },
  V2:  { name: 'V2 (PCA)',  fraudMean: 2.4,  normalMean: 0.0, std: 1.0, importance: 0.06, desc: 'Transaction Type Pattern' },
  Amount: { name: 'Amount ($)', fraudMean: 122.5, normalMean: 88.3, std: 250, importance: 0.05, desc: 'Transaction Amount in USD' }
};

/**
 * Generate synthetic dataset adhering to Kaggle Credit Card Fraud properties
 */
export function generateSyntheticDataset(sampleCount = 5000, fraudRatio = 0.015) {
  const rng = pseudoRandom(42);
  const data = [];
  const fraudCount = Math.max(1, Math.round(sampleCount * fraudRatio));
  const normalCount = sampleCount - fraudCount;

  // Generate Normal Transactions
  for (let i = 0; i < normalCount; i++) {
    const time = Math.floor(rng() * 172800);
    const amount = Math.max(1.0, Math.exp(randomNormal(3.2, 1.2, rng)));
    
    const row = {
      id: `TX-${100000 + i}`,
      Time: time,
      Amount: parseFloat(amount.toFixed(2)),
      Class: 0,
      V14: randomNormal(0.1, 0.9, rng),
      V12: randomNormal(0.05, 0.95, rng),
      V10: randomNormal(0.0, 1.0, rng),
      V4:  randomNormal(0.0, 1.0, rng),
      V11: randomNormal(0.0, 1.0, rng),
      V17: randomNormal(0.0, 1.0, rng),
      V2:  randomNormal(0.0, 1.0, rng),
    };
    data.push(row);
  }

  // Generate Fraud Transactions
  for (let i = 0; i < fraudCount; i++) {
    const time = Math.floor(rng() * 172800);
    const isLarge = rng() > 0.4;
    const amount = isLarge ? randomNormal(350, 180, rng) : randomNormal(12, 5, rng);

    const row = {
      id: `TX-F${900000 + i}`,
      Time: time,
      Amount: parseFloat(Math.max(1.0, amount).toFixed(2)),
      Class: 1,
      V14: randomNormal(-4.5, 1.3, rng),
      V12: randomNormal(-3.8, 1.2, rng),
      V10: randomNormal(-3.2, 1.1, rng),
      V4:  randomNormal(3.6, 1.2, rng),
      V11: randomNormal(2.9, 1.1, rng),
      V17: randomNormal(-3.1, 1.3, rng),
      V2:  randomNormal(2.4, 1.1, rng),
    };
    data.push(row);
  }

  // Shuffle dataset
  for (let i = data.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [data[i], data[j]] = [data[j], data[i]];
  }

  return data;
}

/**
 * Perform Resampling Simulation (SMOTE, Undersampling, Class Weighting)
 */
export function simulateResampling(dataset, method = 'smote') {
  const normal = dataset.filter(d => d.Class === 0);
  const fraud = dataset.filter(d => d.Class === 1);

  if (method === 'raw') {
    return {
      name: 'Baseline (Imbalanced Data)',
      description: 'Original raw data with 0.17% minority fraud instances. Models suffer from high majority bias.',
      normalCount: normal.length,
      fraudCount: fraud.length,
      ratio: (fraud.length / dataset.length * 100).toFixed(2) + '%',
      scatterData: dataset.slice(0, 300)
    };
  }

  if (method === 'smote') {
    const syntheticFraud = [...fraud];
    const targetCount = Math.min(normal.length, fraud.length * 8);
    const needed = targetCount - fraud.length;

    for (let i = 0; i < needed; i++) {
      const parent = fraud[i % fraud.length];
      const neighbor = fraud[(i + 1) % fraud.length];
      const lambda = Math.random();

      const synth = {
        id: `SYNTH-${i}`,
        Time: Math.round(parent.Time + lambda * (neighbor.Time - parent.Time)),
        Amount: parseFloat((parent.Amount + lambda * (neighbor.Amount - parent.Amount)).toFixed(2)),
        Class: 1,
        V14: parent.V14 + lambda * (neighbor.V14 - parent.V14),
        V12: parent.V12 + lambda * (neighbor.V12 - parent.V12),
        V10: parent.V10 + lambda * (neighbor.V10 - parent.V10),
        V4:  parent.V4 + lambda * (neighbor.V4 - parent.V4),
        V11: parent.V11 + lambda * (neighbor.V11 - parent.V11),
        V17: parent.V17 + lambda * (neighbor.V17 - parent.V17),
        V2:  parent.V2 + lambda * (neighbor.V2 - parent.V2),
        isSynthetic: true
      };
      syntheticFraud.push(synth);
    }

    const combined = [...normal, ...syntheticFraud];
    return {
      name: 'SMOTE Oversampling',
      description: 'Synthetic Minority Over-sampling Technique creates synthetic fraud samples in feature space.',
      normalCount: normal.length,
      fraudCount: syntheticFraud.length,
      ratio: (syntheticFraud.length / combined.length * 100).toFixed(2) + '%',
      scatterData: combined.slice(0, 350)
    };
  }

  if (method === 'undersample') {
    const sampledNormal = normal.slice(0, fraud.length * 3);
    const combined = [...sampledNormal, ...fraud];
    return {
      name: 'Random NearMiss Undersampling',
      description: 'Randomly drops majority normal transactions to balance classes. Fast, but discards data.',
      normalCount: sampledNormal.length,
      fraudCount: fraud.length,
      ratio: (fraud.length / combined.length * 100).toFixed(2) + '%',
      scatterData: combined
    };
  }

  if (method === 'class_weight') {
    const weightRatio = (normal.length / fraud.length).toFixed(1);
    return {
      name: 'Cost-Sensitive Class Weighting',
      description: `Applies scale_pos_weight = ${weightRatio} during loss computation without altering raw sample count.`,
      normalCount: normal.length,
      fraudCount: fraud.length,
      ratio: (fraud.length / dataset.length * 100).toFixed(2) + '%',
      scatterData: dataset.slice(0, 300)
    };
  }
}

/**
 * Benchmark Classifier Performance across Resampling strategies
 */
export function getModelPerformanceSpecs(modelKey = 'xgboost', resampling = 'smote', threshold = 0.5) {
  const benchmarks = {
    logistic_regression: {
      raw:         { baseP: 0.85, baseR: 0.62, roc: 0.92, prAuc: 0.72 },
      smote:       { baseP: 0.78, baseR: 0.89, roc: 0.95, prAuc: 0.82 },
      undersample: { baseP: 0.42, baseR: 0.94, roc: 0.93, prAuc: 0.68 },
      class_weight:{ baseP: 0.76, baseR: 0.91, roc: 0.96, prAuc: 0.84 }
    },
    random_forest: {
      raw:         { baseP: 0.94, baseR: 0.76, roc: 0.95, prAuc: 0.84 },
      smote:       { baseP: 0.88, baseR: 0.92, roc: 0.98, prAuc: 0.89 },
      undersample: { baseP: 0.55, baseR: 0.95, roc: 0.95, prAuc: 0.76 },
      class_weight:{ baseP: 0.89, baseR: 0.93, roc: 0.98, prAuc: 0.90 }
    },
    xgboost: {
      raw:         { baseP: 0.95, baseR: 0.81, roc: 0.97, prAuc: 0.88 },
      smote:       { baseP: 0.91, baseR: 0.95, roc: 0.99, prAuc: 0.94 },
      undersample: { baseP: 0.62, baseR: 0.97, roc: 0.97, prAuc: 0.82 },
      class_weight:{ baseP: 0.92, baseR: 0.96, roc: 0.99, prAuc: 0.95 }
    },
    neural_network: {
      raw:         { baseP: 0.88, baseR: 0.70, roc: 0.94, prAuc: 0.79 },
      smote:       { baseP: 0.84, baseR: 0.93, roc: 0.97, prAuc: 0.88 },
      undersample: { baseP: 0.48, baseR: 0.95, roc: 0.94, prAuc: 0.71 },
      class_weight:{ baseP: 0.85, baseR: 0.94, roc: 0.97, prAuc: 0.89 }
    },
    isolation_forest: {
      raw:         { baseP: 0.35, baseR: 0.85, roc: 0.88, prAuc: 0.52 },
      smote:       { baseP: 0.38, baseR: 0.86, roc: 0.89, prAuc: 0.54 },
      undersample: { baseP: 0.30, baseR: 0.88, roc: 0.86, prAuc: 0.48 },
      class_weight:{ baseP: 0.36, baseR: 0.85, roc: 0.88, prAuc: 0.53 }
    }
  };

  const spec = (benchmarks[modelKey] && benchmarks[modelKey][resampling]) 
    ? benchmarks[modelKey][resampling] 
    : benchmarks.xgboost.smote;

  const tDiff = threshold - 0.5;
  let recall = Math.min(0.99, Math.max(0.10, spec.baseR - tDiff * 0.45));
  let precision = Math.min(0.99, Math.max(0.05, spec.baseP + tDiff * 0.35));

  const totalEval = 56962;
  const actualFraud = 98;
  const actualNormal = totalEval - actualFraud;

  const TP = Math.round(actualFraud * recall);
  const FN = actualFraud - TP;
  const FP = Math.round(TP * (1 - precision) / precision);
  const TN = actualNormal - FP;

  const accuracy = (TP + TN) / totalEval;
  const f1 = (2 * precision * recall) / (precision + recall || 1);
  const f2 = (5 * precision * recall) / (4 * precision + recall || 1);

  const costPerFN = 500;
  const costPerFP = 15;
  const totalFinancialCost = (FN * costPerFN) + (FP * costPerFP);
  const baselineNoMLCost = actualFraud * costPerFN;
  const costSavings = baselineNoMLCost - totalFinancialCost;

  return {
    modelKey,
    resampling,
    threshold,
    precision: parseFloat(precision.toFixed(4)),
    recall: parseFloat(recall.toFixed(4)),
    accuracy: parseFloat(accuracy.toFixed(5)),
    f1Score: parseFloat(f1.toFixed(4)),
    f2Score: parseFloat(f2.toFixed(4)),
    rocAuc: spec.roc,
    prAuc: spec.prAuc,
    confusionMatrix: { TP, FN, FP, TN },
    financial: {
      totalFinancialCost,
      baselineNoMLCost,
      costSavings,
      savingsPercent: ((costSavings / baselineNoMLCost) * 100).toFixed(1) + '%'
    }
  };
}

/**
 * Generate ROC Curve points (FPR vs TPR)
 */
export function generateROCCurve(rocAuc = 0.99) {
  const points = [];
  const steps = 25;
  for (let i = 0; i <= steps; i++) {
    const fpr = i / steps;
    const power = (1 - rocAuc) * 12 + 1;
    const tpr = Math.min(1, Math.pow(fpr, 1 / power));
    points.push({
      fpr: parseFloat(fpr.toFixed(3)),
      tpr: parseFloat(tpr.toFixed(3)),
      random: parseFloat(fpr.toFixed(3))
    });
  }
  return points;
}

/**
 * Generate Precision-Recall Curve points (Recall vs Precision)
 */
export function generatePRCurve(prAuc = 0.94) {
  const points = [];
  const steps = 25;
  for (let i = 0; i <= steps; i++) {
    const recall = i / steps;
    const power = (1 - prAuc) * 10 + 1;
    const precision = Math.max(0.02, 1 - Math.pow(recall, power));
    points.push({
      recall: parseFloat(recall.toFixed(3)),
      precision: parseFloat(precision.toFixed(3)),
      baselineRatio: 0.0017
    });
  }
  return points;
}

/**
 * Predict Fraud Probability and generate SHAP-like feature attributions for live simulation
 */
export function predictFraudScore(features) {
  const weights = {
    V14: -0.85,
    V12: -0.72,
    V10: -0.65,
    V4:   0.68,
    V11:  0.58,
    V17: -0.60,
    V2:   0.35,
    Amount: 0.003
  };

  let logit = -3.2;
  const attributions = [];

  for (const [key, prop] of Object.entries(FEATURE_PROPERTIES)) {
    const val = features[key] !== undefined ? parseFloat(features[key]) : prop.normalMean;
    const weight = weights[key] || 0.1;
    const delta = (val - prop.normalMean) * weight;
    logit += delta;

    attributions.push({
      feature: prop.name,
      key,
      value: val,
      contribution: parseFloat(delta.toFixed(3)),
      impact: delta > 0 ? 'Risk Increase' : 'Risk Decrease'
    });
  }

  const probability = 1 / (1 + Math.exp(-logit));
  const roundedProb = parseFloat(probability.toFixed(4));

  let riskLevel = 'LOW';
  let badgeColor = 'emerald';
  if (roundedProb >= 0.75) {
    riskLevel = 'CRITICAL FRAUD';
    badgeColor = 'rose';
  } else if (roundedProb >= 0.40) {
    riskLevel = 'HIGH SUSPICION';
    badgeColor = 'amber';
  } else if (roundedProb >= 0.15) {
    riskLevel = 'MODERATE RISK';
    badgeColor = 'blue';
  }

  attributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

  return {
    probability: roundedProb,
    percentScore: (roundedProb * 100).toFixed(1) + '%',
    riskLevel,
    badgeColor,
    attributions
  };
}

/**
 * Parse CSV text and run batch fraud predictions
 */
export function parseCSVAndPredictBatch(csvText) {
  const lines = csvText.split(/\r\n|\n/).filter(line => line.trim() !== '');
  if (lines.length <= 1) {
    throw new Error('CSV file is empty or missing data rows.');
  }

  const headers = lines[0].split(',').map(h => h.trim().replace(/^["']|["']$/g, ''));
  
  const results = [];
  let totalFraud = 0;
  let totalHighSuspicion = 0;
  let totalModerate = 0;
  let totalSafe = 0;
  let totalDollarAtRisk = 0;

  for (let i = 1; i < lines.length; i++) {
    const rowValues = lines[i].split(',').map(val => val.trim().replace(/^["']|["']$/g, ''));
    if (rowValues.length < headers.length) continue;

    const rowObj = {};
    headers.forEach((h, idx) => {
      const num = parseFloat(rowValues[idx]);
      rowObj[h] = isNaN(num) ? rowValues[idx] : num;
    });

    const txId = rowObj.id || rowObj.ID || rowObj.TransactionId || `CSV-TX-${1000 + i}`;
    const amount = typeof rowObj.Amount === 'number' ? rowObj.Amount : (typeof rowObj.amount === 'number' ? rowObj.amount : 50.0);
    const time = typeof rowObj.Time === 'number' ? rowObj.Time : (typeof rowObj.time === 'number' ? rowObj.time : i * 10);

    const featureInputs = {
      Time: time,
      Amount: amount,
      V14: typeof rowObj.V14 === 'number' ? rowObj.V14 : 0.0,
      V12: typeof rowObj.V12 === 'number' ? rowObj.V12 : 0.0,
      V10: typeof rowObj.V10 === 'number' ? rowObj.V10 : 0.0,
      V4:  typeof rowObj.V4 === 'number' ? rowObj.V4 : 0.0,
      V11: typeof rowObj.V11 === 'number' ? rowObj.V11 : 0.0,
      V17: typeof rowObj.V17 === 'number' ? rowObj.V17 : 0.0,
      V2:  typeof rowObj.V2 === 'number' ? rowObj.V2 : 0.0,
    };

    const pred = predictFraudScore(featureInputs);

    if (pred.riskLevel === 'CRITICAL FRAUD') {
      totalFraud++;
      totalDollarAtRisk += amount;
    } else if (pred.riskLevel === 'HIGH SUSPICION') {
      totalHighSuspicion++;
      totalDollarAtRisk += amount * 0.5;
    } else if (pred.riskLevel === 'MODERATE RISK') {
      totalModerate++;
    } else {
      totalSafe++;
    }

    results.push({
      rowNumber: i,
      id: txId,
      Time: time,
      Amount: amount,
      probability: pred.probability,
      percentScore: pred.percentScore,
      riskLevel: pred.riskLevel,
      badgeColor: pred.badgeColor,
      topRiskFeature: pred.attributions[0]?.feature || 'N/A',
      rawRow: rowObj
    });
  }

  return {
    totalProcessed: results.length,
    totalFraud,
    totalHighSuspicion,
    totalModerate,
    totalSafe,
    totalDollarAtRisk: parseFloat(totalDollarAtRisk.toFixed(2)),
    predictions: results
  };
}

/**
 * Generate sample CSV template string for testing import
 */
export function generateSampleCSVTemplate() {
  return `Time,Amount,V14,V12,V10,V4,V11,V17,V2
0,42.50,0.15,-0.08,0.02,0.1,-0.2,0.05,-0.1
120,1450.00,-4.85,-3.9,-3.1,3.4,2.8,-3.3,2.6
350,1.25,-5.2,-4.1,-4.5,4.1,3.2,-4.0,3.1
540,88.20,0.05,0.01,-0.02,-0.1,0.05,0.02,0.1
820,3200.00,0.2,0.1,0.0,-0.3,-0.1,0.1,-0.2
990,195.00,-4.1,-3.2,-2.9,2.8,2.5,-3.0,2.1
1200,15.99,0.08,0.02,0.0,0.0,-0.1,0.0,0.0`;
}

/**
 * Preset Transactions for Live Simulator
 */
export const PRESET_TRANSACTIONS = [
  {
    name: 'Normal Grocery Store Purchase',
    description: 'Standard domestic point-of-sale transaction with typical PCA readings.',
    features: { V14: 0.15, V12: -0.08, V10: 0.02, V4: 0.1, V11: -0.2, V17: 0.05, V2: -0.1, Amount: 42.50, Time: 45000 }
  },
  {
    name: 'Suspicious High-Amount Drain',
    description: 'Abnormal V14 drop (-4.8) paired with high transaction amount ($1,450).',
    features: { V14: -4.85, V12: -3.9, V10: -3.1, V4: 3.4, V11: 2.8, V17: -3.3, V2: 2.6, Amount: 1450.00, Time: 82100 }
  },
  {
    name: 'Bot Account Probe Charge',
    description: 'Tiny $1.25 micro-charge with extreme PCA anomaly profile characteristic of automated card testing.',
    features: { V14: -5.2, V12: -4.1, V10: -4.5, V4: 4.1, V11: 3.2, V17: -4.0, V2: 3.1, Amount: 1.25, Time: 12000 }
  },
  {
    name: 'Legitimate International Luxury Purchase',
    description: 'High transaction amount ($3,200) but standard PCA feature profile.',
    features: { V14: 0.2, V12: 0.1, V10: 0.0, V4: -0.3, V11: -0.1, V17: 0.1, V2: -0.2, Amount: 3200.00, Time: 64000 }
  }
];
