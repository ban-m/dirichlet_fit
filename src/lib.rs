#[macro_use]
extern crate log;

#[link(name = "m")]
extern "C" {
    fn lgamma(x: f64) -> f64;
}

/// Return log Dir(`probability`|`parameters`).
/// # Panics
/// Panics if an element of the `probability` is non-positive.
/// Panics if `probability` does not sum up to 1.
/// Panics if an element of the `parameters` is non-positive.
/// Panics if `probability.len() != parameters.len()`.
/// # Example
/// ```rust
/// let prob = [0.1,0.9];
/// let param = [10f64,90f64];
/// let likelihood = dirichlet_fit::dirichlet(&prob,&param);
/// assert!((likelihood.exp()-13.186534).abs() < 0.001);
/// ```
pub fn dirichlet(probability: &[f64], parameters: &[f64]) -> f64 {
    assert_eq!(probability.len(), parameters.len());
    assert!(probability.iter().all(|x| x.is_sign_positive()));
    assert!(parameters.iter().all(|x| x.is_sign_positive()));
    let sum_prob: f64 = probability.iter().sum();
    assert!((1f64 - sum_prob).abs() < 0.0001);
    let mut sum = 0f64;
    // Scaling factor.
    let mut scale = 0f64;
    // Likelihood depends on `probability`
    let mut lk = 0f64;
    for (&param, prob) in parameters.iter().zip(probability.iter()) {
        sum += param;
        scale += unsafe { lgamma(param) };
        lk += (param - 1f64) * prob.ln();
    }
    unsafe { lgamma(sum) - scale + lk }
}

/// Return log Dir(exp(`log_prob`)|`parameters`).
/// The difference between [dirichlet] is that it takes the input probability as loged form.
/// # Example
/// ```rust
/// let prob = [0.1f64.ln(),0.9f64.ln()]; // <- same as [0.1,0.9].
/// let param = [10f64,90f64];
/// let likelihood = dirichlet_fit::dirichlet_log(&prob,&param);
/// assert!((likelihood.exp()-13.186534).abs() < 0.001);
/// ```
pub fn dirichlet_log(log_prob: &[f64], parameters: &[f64]) -> f64 {
    let mut sum = 0f64;
    // Scaling factor.
    let mut scale = 0f64;
    // Likelihood depends on `probability`
    let mut lk = 0f64;
    for (&param, log_prob) in parameters.iter().zip(log_prob.iter()) {
        sum += param;
        scale += unsafe { lgamma(param) };
        lk += (param - 1f64) * log_prob;
    }
    unsafe { lgamma(sum) - scale + lk }
}

/// Return the estimated parameters maximixing `Dir(exp(data)|parameters)`
/// Make sure that the input data shoule be logarithmic version of the probability.
/// # Panics
/// Panics if a datum has a non-negative element including zero (i.e., log prob < 0).
/// Panics if a datum does not sum up to 1 when exped.
/// Panics if data do not share the same length.
/// Panics if data is empty.
pub fn fit<D: std::borrow::Borrow<[f64]>>(data: &[D]) -> Vec<f64> {
    let weights = vec![(1f64, vec![1f64; data.len()])];
    fit_multiple(&[data], &weights)
}

/// Return the estimated parameters maximixing the "weighted"-likelihood defined as follows:
/// ```ignore
/// let mut lk:f64 = 0f64;
/// for (ds, (w_data,ws)) in dataset.iter().zip(weights.iter()){
///     lk += w_data * ds.iter().zip(ws.iter()).map(|(d,w)|w * dirichlet_log(d,&parameters)).sum::<f64>();
/// }
/// ```
/// # Panics
/// Panics if an element in the dataset panics `dirichlet(data, param)`.
pub fn fit_multiple<
    D: std::borrow::Borrow<[f64]>,
    E: std::borrow::Borrow<[D]>,
    W: std::borrow::Borrow<[f64]>,
>(
    dataset: &[E],
    weights: &[(f64, W)],
) -> Vec<f64> {
    if let Some(res) = validate(dataset, weights) {
        return res;
    };
    let dim = dataset[0].borrow()[0].borrow().len();
    let mut parameters = vec![0f64; dim];
    for (ds, (w_data, ws)) in dataset.iter().zip(weights.iter()) {
        for (xs, w) in ds.borrow().iter().zip(ws.borrow().iter()) {
            for (p, x) in parameters.iter_mut().zip(xs.borrow().iter()) {
                *p += w_data * w * x.exp();
            }
        }
    }
    let mut optimizer: AdamOptimizer = Optimizer::new(dim);
    fit_multiple_with(dataset, weights, &mut optimizer, &parameters)
}

// Validate input.
fn validate<
    D: std::borrow::Borrow<[f64]>,
    E: std::borrow::Borrow<[D]>,
    W: std::borrow::Borrow<[f64]>,
>(
    dataset: &[E],
    weights: &[(f64, W)],
) -> Option<Vec<f64>> {
    assert_eq!(dataset.len(), weights.len());
    let dim = dataset[0].borrow()[0].borrow().len();
    if dim == 1 {
        warn!("The input data is 1-dim probability simplex, {{1.0}}.");
        warn!("All 1D dirichlet distribution Dir(x|a) is 1. ML estimator is meaningless.");
        warn!("Just retrun 1.");
        return Some(vec![1f64]);
    }
    for (ds, &(w_data, ref ws)) in dataset.iter().zip(weights.iter()) {
        assert!(w_data <= 1f64);
        assert!(w_data >= 0f64);
        let ws = ws.borrow();
        let ds: Vec<_> = ds.borrow().iter().map(|xs| xs.borrow()).collect();
        assert_eq!(ds.len(), ws.len());
        for xs in ds.iter() {
            assert_eq!(xs.len(), dim);
            let sum: f64 = xs.iter().map(|x| x.exp()).sum();
            assert!((1f64 - sum).abs() < 0.1);
            xs.iter().for_each(|x| assert!(x.is_sign_negative()));
        }
    }
    None
}

/// Return the estimated parameters maximixing the "weighted"-likelihood defined as follows:
/// ```ignore
/// let mut lk:f64 = 0f64;
/// for (ds, (w_data,ws)) in dataset.iter().zip(weights.iter()){
///     lk += w_data * ds.iter().zip(ws.iter()).map(|(d,w)|w * dirichlet_log(d,&parameters)).sum::<f64>();
/// }
/// ```
/// # Panics
/// Panics if an element in the dataset panics `dirichlet(data, param)`.
/// You can handle the detail of the optimizing method, including of the parameters used in the optimizer,
/// the initial value of the parameters, and additional constraint on the norm of the parameter(L2).
pub fn fit_multiple_with<
    D: std::borrow::Borrow<[f64]>,
    E: std::borrow::Borrow<[D]>,
    W: std::borrow::Borrow<[f64]>,
    O: Optimizer,
>(
    dataset: &[E],
    weights: &[(f64, W)],
    optimizer: &mut O,
    initial_param: &[f64],
) -> Vec<f64> {
    if let Some(res) = validate(dataset, weights) {
        return res;
    };
    let dim = dataset[0].borrow()[0].borrow().len();
    assert_eq!(initial_param.len(), dim);
    let mut dataset: Vec<_> = dataset
        .iter()
        .zip(weights.iter())
        .map(|(ds, (w_data, ws))| {
            let ws = ws.borrow();
            let ds: Vec<_> = ds.borrow().iter().map(|xs| xs.borrow()).collect();
            (ds, (*w_data, ws))
        })
        .collect();
    let mut parameters = initial_param.to_vec();
    optimizer.optim(&mut dataset, &mut parameters);
    parameters
}

use rand::prelude::SliceRandom;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

fn likelihood(data: &[(Vec<&[f64]>, (f64, &[f64]))], param: &[f64]) -> f64 {
    data.iter()
        .map(|(ds, (w_data, ws))| -> f64 {
            let lk_data: f64 = ds
                .iter()
                .zip(ws.iter())
                .map(|(prob, w)| w * dirichlet_log(&prob, param))
                .sum();
            lk_data * w_data
        })
        .sum()
}

/// If the liklihood did not increase `STOP_COUNT` time, it early stops.
pub const STOP_COUNT: usize = 10;
pub trait Optimizer {
    fn new(dim: usize) -> Self;
    fn tik(&mut self);
    fn update(&mut self, ds: &[&[f64]], weights: (f64, &[f64]), param: &mut [f64]);
    fn loop_count(&self) -> usize;
    fn threshold(&self) -> f64;
    fn shuffle(&mut self, data: &mut [(Vec<&[f64]>, (f64, &[f64]))]);
    fn normalize(&self, param: &mut [f64]) {
        let sum: f64 = param.iter().map(|x| x * x).sum::<f64>().sqrt();
        param.iter_mut().for_each(|x| *x /= sum);
        if let Some(norm) = self.norm() {
            param.iter_mut().for_each(|x| *x *= norm)
        }
    }
    /// If there are some L2 constraint on the norm,
    /// return Some(norm), otherwise, none.
    fn norm(&self) -> Option<f64> {
        None
    }
    fn optim(&mut self, data: &mut [(Vec<&[f64]>, (f64, &[f64]))], param: &mut [f64]) {
        // trace!("GRAD\tCount\tGrad");
        // trace!("LK\tCount\tLoss");
        let mut lk = likelihood(data, param);
        // trace!("LK\t{}\t{:.3}", self.loop_count(), lk);
        let mut count_not_increased = 0;
        for _ in 0..10000 {
            self.shuffle(data);
            for (ds, weight) in data.iter() {
                self.update(ds.as_slice(), *weight, param);
            }
            // Rounding...
            self.normalize(param);
            let likelihood = likelihood(data, param);
            self.tik();
            // trace!("LK\t{}\t{:.3}", self.loop_count(), likelihood,);
            // trace!("PARAM\t{}\t[{}]", self.loop_count(), vec2str(&param));
            if likelihood <= lk {
                count_not_increased += 1;
            } else {
                count_not_increased = 0;
            }
            lk = lk.max(likelihood);
            if STOP_COUNT < count_not_increased {
                break;
            }
        }
    }
}

#[allow(dead_code)]
/// Adam Optimizer.
pub struct AdamOptimizer {
    dim: usize,
    loop_count: usize,
    lr: f64,
    beta_1: f64,
    beta_2: f64,
    epsilon: f64,
    decay: f64,
    moment: Vec<f64>,
    sq_moment: Vec<f64>,
    rng: Xoshiro256StarStar,
    threshold: f64,
    norm: Option<f64>,
}

impl AdamOptimizer {
    /// # Arguments
    /// - `dim`: Dimension of the Dirichlet distribution to be estimated.
    /// - `learning_rate`: Learning rate of the Adam. Usually near 0, Recommend 0.01~0.03.
    /// - `beta_1`: Decay rate of the 1st order moment. Recommend 0.9.
    /// - `beta_2`: Decay rate of the 2nd order moment. Recommend 0.999.
    /// - `decay`: decay rate of the learinng rate. Recommend 1.
    /// - `seed`: seed used in the random number generator in AdamOptimizer.
    /// - `threshold`: if the lielihood did not increased `threshold` for T(=10) times, it early drops.
    ///
    pub fn with_details(
        dim: usize,
        learning_rate: f64,
        (beta_1, beta_2): (f64, f64),
        decay: f64,
        threshold: f64,
        norm: Option<f64>,
        seed: u64,
    ) -> Self {
        Self {
            dim,
            loop_count: 0,
            lr: learning_rate,
            threshold,
            beta_1,
            beta_2,
            epsilon: 0.00000001,
            decay,
            moment: vec![0f64; dim],
            sq_moment: vec![0f64; dim],
            rng: SeedableRng::seed_from_u64(seed),
            norm,
        }
    }
    /// Change learning rate.
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }
    /// Set normalize constraint.
    pub fn norm(mut self, norm: f64) -> Self {
        self.norm = Some(norm);
        self
    }
}

impl Optimizer for AdamOptimizer {
    fn threshold(&self) -> f64 {
        self.threshold
    }
    fn norm(&self) -> Option<f64> {
        self.norm
    }
    fn new(dim: usize) -> Self {
        Self {
            dim,
            loop_count: 0,
            lr: 0.01,
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 0.0000001,
            decay: 0.995,
            threshold: 0.0001,
            moment: vec![0f64; dim],
            sq_moment: vec![0f64; dim],
            rng: SeedableRng::seed_from_u64(34820),
            norm: None,
        }
    }
    fn tik(&mut self) {
        self.lr *= self.decay;
        self.loop_count += 1;
    }
    fn update(&mut self, ds: &[&[f64]], (w_data, ws): (f64, &[f64]), param: &mut [f64]) {
        // Calculate gradient.
        let grad = get_gradient(ds, w_data, ws, param);
        // If normalization is on, calc orthogonal vector.
        let grad = match self.norm {
            Some(_) => to_orthogonal(grad, param),
            None => grad,
        };
        // Update moment, sq_moment.
        self.moment
            .iter_mut()
            .zip(grad.iter())
            .for_each(|(m, g)| *m = self.beta_1 * *m + (1.0 - self.beta_1) * g);
        self.sq_moment
            .iter_mut()
            .zip(grad.iter())
            .for_each(|(v, g)| *v = self.beta_2 * *v + (1.0 - self.beta_2) * g * g);
        // calc gradient.
        let factor_1 = 1f64 - self.beta_1.powi(self.loop_count as i32 + 1);
        let factor_2 = 1f64 - self.beta_2.powi(self.loop_count as i32 + 1);
        let grad: Vec<_> = self.moment.iter().map(|m| m / factor_1).collect();
        let var: Vec<_> = self.sq_moment.iter().map(|v| v / factor_2).collect();
        let grad: Vec<_> = grad
            .into_iter()
            .zip(var.iter())
            .map(|(g, v)| {
                let div = v.sqrt() + self.epsilon;
                self.lr * g / div
            })
            .collect();
        // Scaling...
        let scaler = grad
            .iter()
            .zip(param.iter())
            .filter(|&(g, p)| p + g < 0f64)
            .map(|(g, p)| 2.0 * g.abs() / p)
            .max_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap_or(1f64);
        trace!(
            "GRAD\t{}\t{}\t{}\t{}",
            self.loop_count(),
            vec2str(&param),
            vec2str(&grad),
            scaler,
        );
        param
            .iter_mut()
            .zip(grad)
            .for_each(|(p, g)| *p += g / scaler);
    }
    fn loop_count(&self) -> usize {
        self.loop_count
    }
    fn shuffle(&mut self, data: &mut [(Vec<&[f64]>, (f64, &[f64]))]) {
        data.shuffle(&mut self.rng);
    }
}

#[allow(dead_code)]
pub struct MomentumOptimizer {
    dim: usize,
    loop_count: usize,
    lr: f64,
    // Decay rate of the momentum
    decay: f64,
    moment: Vec<f64>,
    rng: Xoshiro256StarStar,
    threshold: f64,
}

impl Optimizer for MomentumOptimizer {
    fn threshold(&self) -> f64 {
        self.threshold
    }
    fn shuffle(&mut self, data: &mut [(Vec<&[f64]>, (f64, &[f64]))]) {
        data.shuffle(&mut self.rng);
    }
    fn new(dim: usize) -> Self {
        Self {
            dim,
            loop_count: 0,
            lr: 0.01,
            decay: 0.9,
            threshold: 0.0001,
            moment: vec![0f64; dim],
            rng: SeedableRng::seed_from_u64(432543543),
        }
    }
    fn tik(&mut self) {
        self.loop_count += 1;
    }
    fn update(&mut self, ds: &[&[f64]], (w_data, ws): (f64, &[f64]), param: &mut [f64]) {
        // Calc crad
        let grad = get_gradient(ds, w_data, ws, param);
        // Merge with current moment.
        let lr = self.lr * (1f64 - self.decay);
        self.moment
            .iter_mut()
            .zip(grad)
            .for_each(|(m, g)| *m = self.decay * *m + lr * g);
        // Reduce if the moment is too large.
        if let Some(scale) = self
            .moment
            .iter()
            .zip(param.iter())
            .filter(|&(g, p)| p + g < 0f64)
            .map(|(g, p)| p / 2.0 / g)
            .min_by(|x, y| x.partial_cmp(y).unwrap())
        {
            self.moment.iter_mut().for_each(|x| *x /= scale);
        }
        let scalar: f64 = self.moment.iter().map(|x| x * x).sum();
        trace!(
            "GRAD\t{}\t{:.2}\t[{}]",
            self.loop_count(),
            scalar,
            vec2str(&self.moment)
        );
        param
            .iter_mut()
            .zip(&self.moment)
            .for_each(|(p, m)| *p += m);
    }
    fn loop_count(&self) -> usize {
        self.loop_count
    }
}

#[allow(dead_code)]
struct SGDOptimizer {
    dim: usize,
    loop_count: usize,
    // Learning rate.
    lr: f64,
    // learning decay rate.
    decay: f64,
    rng: Xoshiro256StarStar,
    threshold: f64,
}

impl Optimizer for SGDOptimizer {
    fn threshold(&self) -> f64 {
        self.threshold
    }
    fn shuffle(&mut self, data: &mut [(Vec<&[f64]>, (f64, &[f64]))]) {
        data.shuffle(&mut self.rng);
    }
    fn loop_count(&self) -> usize {
        self.loop_count
    }
    fn tik(&mut self) {
        self.loop_count += 1;
        self.lr *= self.decay;
    }
    fn new(dim: usize) -> Self {
        Self {
            loop_count: 0,
            dim,
            lr: 0.01,
            decay: 0.999,
            threshold: 0.00001,
            rng: SeedableRng::seed_from_u64(232342),
        }
    }
    // Return likelihood.
    fn update(&mut self, ds: &[&[f64]], (w_data, ws): (f64, &[f64]), param: &mut [f64]) {
        // Calculate gradient.
        let mut grad = get_gradient(ds, w_data, ws, param);
        grad.iter_mut().for_each(|x| *x *= self.lr);
        // Scaling...
        if let Some(scale) = grad
            .iter()
            .zip(param.iter())
            .filter(|&(g, p)| p + g < 0f64)
            .map(|(g, p)| p / 2.0 / g)
            .max_by(|x, y| x.partial_cmp(y).unwrap())
        {
            grad.iter_mut().for_each(|x| *x /= scale);
        }
        let scalar: f64 = grad.iter().map(|x| x * x).sum();
        trace!(
            "GRAD\t{}\t{:.2}\t[{}]",
            self.loop_count(),
            scalar,
            vec2str(&grad)
        );
        param.iter_mut().zip(grad.iter()).for_each(|(p, g)| *p += g);
    }
}

fn vec2str(xs: &[f64]) -> String {
    let xs: Vec<_> = xs.iter().map(|x| format!("{:.2}", x)).collect();
    xs.join(",")
}

fn get_gradient(ds: &[&[f64]], w_data: f64, ws: &[f64], param: &[f64]) -> Vec<f64> {
    let param_sum: f64 = param.iter().sum();
    let sum_digam: f64 = digamma(param_sum);
    let weight_sum: f64 = w_data * ws.iter().sum::<f64>();
    let mut grad: Vec<_> = param
        .iter()
        .map(|&p| weight_sum * (sum_digam - digamma(p)))
        .collect();
    for (w, logprob) in ws.iter().zip(ds.iter()) {
        for (grad, x) in grad.iter_mut().zip(logprob.iter()) {
            *grad += w * x;
        }
    }
    grad
}

// Return the component of the `grad` which is orthogonal to the `param`.
fn to_orthogonal(mut grad: Vec<f64>, param: &[f64]) -> Vec<f64> {
    let norm: f64 = param.iter().map(|x| x * x).sum();
    let inner_product: f64 = grad.iter().zip(param.iter()).map(|(x, y)| x * y).sum();
    grad.iter_mut()
        .zip(param.iter())
        .for_each(|(g, p)| *g -= inner_product * p / norm);
    grad
}

// Digamma function. It depends on the offset method + polynomial seriaze approx.
fn digamma(x: f64) -> f64 {
    if x < 10f64 {
        let offset: f64 = (0..10).map(|i| (i as f64 + x).recip()).sum();
        digamma_large(x + 10f64) - offset
    } else {
        digamma_large(x)
    }
}

// Digamma function. x -> d/dx ln Gamma(x).
fn digamma_large(x: f64) -> f64 {
    let mut digam_val = x.ln();
    digam_val -= (2.0 * x).recip();
    digam_val -= (12.0 * x.powi(2)).recip();
    digam_val += (120.0 * x.powi(4)).recip();
    digam_val -= (252.0 * x.powi(6)).recip();
    digam_val += (240.0 * x.powi(8)).recip();
    digam_val -= (132.0 * x.powi(10)).recip();
    digam_val += 691.0 / (32760.0 * x.powi(12));
    digam_val -= (12.0 * x.powi(14)).recip();
    digam_val
}

#[cfg(test)]
pub mod tests {
    use super::*;
    #[test]
    fn it_works() {}
    #[test]
    fn dirichlet_test() {
        let prob = [0.1, 0.9];
        let param = [10f64, 90f64];
        let likelihood = dirichlet(&prob, &param);
        assert!(
            (likelihood.exp() - 13.186534).abs() < 0.001,
            "{}",
            likelihood
        );
        let prob = [0.1; 10];
        let param = vec![vec![10f64; 5], vec![4f64; 5]].concat();
        let lk = dirichlet(&prob, &param);
        assert!((lk.exp() - 3497321.67407045f64).abs() < 0.00001);
    }
    #[test]
    fn digamma_test() {
        fn run_test(x: f64) {
            let diff = 0.000000001f64.min(x / 4f64);
            let back = unsafe { lgamma(x - diff) };
            let forward = unsafe { lgamma(x + diff) };
            let grad = (forward - back) / 2f64 / diff;
            let grad_app = digamma(x);
            let diff = (grad - grad_app).abs();
            assert!(diff < 0.00001, "{}", diff);
        }
        let small = 0.001;
        let len = 10000;
        for test in (0..len).map(|i| i as f64 / len as f64 + small) {
            println!("{}", test);
            run_test(test);
        }
        for test in (10..1000).map(|x| 10f64 + x as f64 / 100f64) {
            println!("{}", test);
            run_test(test);
        }
    }
    #[test]
    fn optim_test_single() {}
}
