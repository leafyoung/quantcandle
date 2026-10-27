//! Port of `torchquantlib/models/stochastic_volatility/gans.py`.
//!
//! The Python file is purely a GAN training harness: `Generator` /
//! `Discriminator` are `nn.Sequential` MLPs and `train_gan` is an
//! Adam/BCELoss training loop (rule 7 — no reusable pricing/simulation
//! numeric core, torch optimizers have no Burn equivalent to port).
//!
//! The numeric pieces it *does* contribute are ported and tested below:
//! the layer shapes/activations of the two networks (evaluated with
//! caller-supplied weights, since torch's random init is not part of the
//! contract) and the BCE loss used by the training loop.

/// Dense layer: `y = x @ W^T + b`, weights row-major (out_features x
/// in_features), matching `nn.Linear`.
pub fn linear(x: &[f64], weight: &[Vec<f64>], bias: &[f64]) -> Vec<f64> {
    weight
        .iter()
        .zip(bias)
        .map(|(row, b)| row.iter().zip(x).map(|(w, xi)| w * xi).sum::<f64>() + b)
        .collect()
}

/// ReLU activation.
pub fn relu(x: &[f64]) -> Vec<f64> {
    x.iter().map(|v| v.max(0.0)).collect()
}

/// `nn.LeakyReLU(0.2)`.
pub fn leaky_relu(x: &[f64], negative_slope: f64) -> Vec<f64> {
    x.iter()
        .map(|v| if *v >= 0.0 { *v } else { negative_slope * *v })
        .collect()
}

/// `nn.Tanh`.
pub fn tanh(x: &[f64]) -> Vec<f64> {
    x.iter().map(|v| v.tanh()).collect()
}

/// `nn.Sigmoid`.
pub fn sigmoid(x: &[f64]) -> Vec<f64> {
    x.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect()
}

/// `Generator` architecture: Linear(in,128) -> ReLU -> Linear(128,256) ->
/// ReLU -> Linear(256,out) -> Tanh, with caller-supplied weights.
pub fn generator_forward(x: &[f64], w1: &[Vec<f64>], b1: &[f64], w2: &[Vec<f64>], b2: &[f64], w3: &[Vec<f64>], b3: &[f64]) -> Vec<f64> {
    let h1 = relu(&linear(x, w1, b1));
    let h2 = relu(&linear(&h1, w2, b2));
    tanh(&linear(&h2, w3, b3))
}

/// `Discriminator` architecture: Linear(in,256) -> LeakyReLU(0.2) ->
/// Linear(256,128) -> LeakyReLU(0.2) -> Linear(128,1) -> Sigmoid.
pub fn discriminator_forward(x: &[f64], w1: &[Vec<f64>], b1: &[f64], w2: &[Vec<f64>], b2: &[f64], w3: &[Vec<f64>], b3: &[f64]) -> f64 {
    let h1 = leaky_relu(&linear(x, w1, b1), 0.2);
    let h2 = leaky_relu(&linear(&h1, w2, b2), 0.2);
    sigmoid(&linear(&h2, w3, b3))[0]
}

/// `nn.BCELoss` reduced by mean, as used in `train_gan` (`criterion`).
pub fn bce_loss(predictions: &[f64], targets: &[f64]) -> f64 {
    let n = predictions.len() as f64;
    predictions
        .iter()
        .zip(targets)
        .map(|(&p, &t)| -(t * p.max(1e-12).ln() + (1.0 - t) * (1.0 - p).max(1e-12).ln()))
        .sum::<f64>()
        / n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1e-300)
    }

    /// Deterministic forward passes with fixed weights; reference from
    /// scripts/gen_gans.py (numpy float64 replication of the architectures),
    /// tol 1e-12 relative.
    #[test]
    fn generator_forward_matches_reference() {
        let x = [0.5, -1.25, 2.0];
        let w1 = [[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]];
        let w1v: Vec<Vec<f64>> = w1.iter().map(|r| r.to_vec()).collect();
        let b1 = [0.05, -0.05];
        let w2 = [[1.0, -1.0], [0.25, 0.75], [-0.5, 2.0]];
        let w2v: Vec<Vec<f64>> = w2.iter().map(|r| r.to_vec()).collect();
        let b2 = [0.1, 0.0, -0.1];
        let w3 = [[0.3, -0.4, 0.5], [0.6, 0.7, -0.8]];
        let w3v: Vec<Vec<f64>> = w3.iter().map(|r| r.to_vec()).collect();
        let b3 = [0.02, -0.02];
        let y = generator_forward(&x, &w1v, &b1, &w2v, &b2, &w3v, &b3);
        let expected = [0.23549574953849794, 0.6505490325113644];
        for (got, want) in y.iter().zip(expected) {
            assert!(rel(*got, want) < 1e-12, "got={got} want={want}");
        }
    }

    #[test]
    fn discriminator_forward_matches_reference() {
        let x = [0.3, -0.7];
        let w1 = [[0.2, 0.4], [-0.1, 0.3], [0.5, -0.25]];
        let w1v: Vec<Vec<f64>> = w1.iter().map(|r| r.to_vec()).collect();
        let b1 = [0.0, 0.1, -0.1];
        let w2 = [[0.3, -0.2, 0.6], [0.1, 0.9, -0.4]];
        let w2v: Vec<Vec<f64>> = w2.iter().map(|r| r.to_vec()).collect();
        let b2 = [0.05, -0.05];
        let w3 = [[0.7, -0.3]];
        let w3v: Vec<Vec<f64>> = w3.iter().map(|r| r.to_vec()).collect();
        let b3 = [0.01];
        let d = discriminator_forward(&x, &w1v, &b1, &w2v, &b2, &w3v, &b3);
        assert!(rel(d, 0.5360264598045577) < 1e-12, "d={d}");
    }

    /// BCE loss vs the torch `nn.BCELoss` reference, tol 1e-12.
    #[test]
    fn bce_loss_matches_reference() {
        let preds = [0.9, 0.2, 0.7];
        let targets = [1.0, 0.0, 1.0];
        let loss = bce_loss(&preds, &targets);
        assert!(rel(loss, 0.22839300363692283) < 1e-12, "loss={loss}");
    }
}
