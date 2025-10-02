# PredNet Brightness Experiments

## Question 1: Does Brightness Penalty Actually Help?

**Hypothesis:** Maybe MSE loss alone is sufficient, and brightness penalty is redundant.

### Experiment Setup:

**Control (Current):**
```python
use_mse_loss = True
mse_weight = 2.0
use_brightness_penalty = True
brightness_weight = 1.0
```
Result: Brightness 0.28 → 0.000027 ✅

**Test:**
```python
use_mse_loss = True
mse_weight = 2.0
use_brightness_penalty = False  # DISABLE
brightness_weight = 0.0
```

### Expected Outcomes:

**If brightness penalty helps:**
- Final brightness metric will be higher (predictions darker)
- Training may take more epochs to converge
- Predictions might be slightly dimmer

**If MSE alone is sufficient:**
- Similar final brightness metric
- Similar prediction quality
- Brightness penalty was redundant

### How to Run:
```bash
# Edit debug.py line 30
use_brightness_penalty = False

# Run
python debug.py

# Compare final brightness value at epoch 10
```

---

## Question 2: Should PredNet Use MSE Errors Instead of L1?

**Hypothesis:** MSE errors (squared) might provide stronger gradients than L1 (absolute) errors.

### Current PredNet Error (L1-like):
```python
E_pos = relu(Ahat - A)  # Positive errors
E_neg = relu(A - Ahat)  # Negative errors
E = cat([E_neg, E_pos])

# Gradient: constant magnitude (doesn't scale with error size)
```

### Proposed MSE Error:
```python
diff = Ahat - A
E_pos = relu(diff) * relu(diff)  # Squared positive errors
E_neg = relu(-diff) * relu(-diff)  # Squared negative errors
E = cat([E_neg, E_pos])

# Gradient: proportional to error size (stronger for large errors)
```

### Pros of MSE Errors:

1. **Stronger gradients for large errors**
   - L1: gradient = ±1 (constant)
   - MSE: gradient = ±2×error (scales with error)

2. **Consistent with MSE reconstruction loss**
   - Both use quadratic penalties
   - Unified optimization objective

3. **May converge faster**
   - Larger errors get corrected more aggressively

### Cons of MSE Errors:

1. **Changes core PredNet design**
   - Original paper uses L1-like errors for temporal prediction
   - May affect layer-wise representation learning

2. **Less robust to outliers**
   - MSE heavily penalizes outliers
   - Could cause instability

3. **Different error representation semantics**
   - E channels represent "how wrong" (magnitude)
   - Squaring changes the meaning of these channels

### How to Test:

**Step 1:** Modify `prednet.py` line 200-210:
```python
# Comment out Option 1 (L1):
# E_pos = self.relu(Ahat - A)
# E_neg = self.relu(A - Ahat)

# Uncomment Option 2 (MSE):
diff = Ahat - A
E_pos = self.relu(diff) * self.relu(diff)
E_neg = self.relu(-diff) * self.relu(-diff)
```

**Step 2:** Run training:
```bash
python debug.py
```

**Step 3:** Compare:
- Training curves (Error, MSE, Brightness)
- Final prediction quality
- Convergence speed

### Expected Outcomes:

**If MSE errors help:**
- Faster convergence (fewer epochs to reach good quality)
- Stronger temporal consistency
- Possibly better layer representations

**If MSE errors hurt:**
- Training instability
- Worse prediction quality
- Overpenalizing small errors in higher layers

---

## Recommended Testing Order:

### Test 1: Brightness Penalty Necessity
**Quick test (~10 min):**
```bash
# debug.py
use_brightness_penalty = False

python debug.py
```

**Look for:**
- Final brightness metric at epoch 10
- Visual prediction quality
- Compare: 0.000027 (with) vs ??? (without)

### Test 2: MSE Errors in PredNet
**More involved (~10 min):**
```bash
# prednet.py lines 200-210
# Swap Option 1 → Option 2

python debug.py
```

**Look for:**
- Training stability
- Final error values
- Temporal prediction quality
- Check if digits are tracked correctly

---

## My Predictions:

### Brightness Penalty:
**Prediction:** MSE alone might be ~80% sufficient, but brightness penalty provides the **final 20% push** to perfectly match mean brightness.

**Why:** MSE optimizes pixel-wise differences, which includes brightness. But brightness penalty adds an **explicit global constraint** that ensures mean brightness matches exactly, not just locally.

**Test result expectation:** Without brightness penalty:
- Final brightness metric: ~0.0001-0.0005 (vs 0.000027 with it)
- Predictions: slightly dimmer but still good
- **Verdict: Helpful but not critical**

### MSE Errors:
**Prediction:** MSE errors might help slightly for **brightness/contrast** but could **hurt temporal prediction quality**.

**Why:** 
- PredNet's temporal prediction relies on error representations flowing between layers
- Squaring errors changes the semantics of what E channels represent
- Higher layers might get overly large error signals

**Test result expectation:**
- Faster initial convergence
- Possibly unstable in later epochs
- May lose temporal tracking quality
- **Verdict: Probably not worth it - original L1 is designed for PredNet's hierarchical temporal prediction**

---

## Alternative: Hybrid Approach

Instead of changing PredNet's core error, you could:

1. **Keep L1 errors for PredNet** (temporal prediction)
2. **Add MSE loss** for reconstruction (already doing this! ✅)
3. **Add brightness penalty** for explicit mean matching (already doing this! ✅)

This gives you:
- L1's robustness for temporal prediction
- MSE's strong gradients for pixels
- Brightness penalty's explicit constraint

**This is actually your current working solution!** 🎉

---

## Summary

| Loss Component | Purpose | Gradient Type | When It Helps |
|---------------|---------|---------------|---------------|
| PredNet Error (L1) | Temporal prediction | Constant | Tracking motion |
| MSE Loss | Pixel reconstruction | Proportional | Matching brightness/structure |
| Brightness Penalty | Global brightness | Explicit | Ensuring not dark |

**Your current config is likely optimal!** But testing will confirm:
- How much brightness penalty contributes
- Whether MSE errors would help or hurt

