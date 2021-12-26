#[macro_use]
extern crate log;
const SMALL_VAL: f64 = 0.0000000000000000000000000000001;
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
/// If you would calculate diriclet log many times, consider using the `Dirichlet` struct.
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

/// Dirichlet distribution.
#[derive(Debug, Clone)]
pub struct Dirichlet {
    dim: usize,
    param: Vec<f64>,
    // Normalize coefficient.
    norm_coef: f64,
}

impl Dirichlet {
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn lk(&self, log_prob: &[f64]) -> f64 {
        match self.dim {
            1 => 0f64,
            _ => {
                assert_eq!(self.param.len(), log_prob.len());
                let lk: f64 = self
                    .param
                    .iter()
                    .zip(log_prob)
                    .map(|(pa, pr)| (pa - 1f64) * pr)
                    .sum();
                self.norm_coef + lk
            }
        }
    }
    pub fn update(&mut self, xs: &[f64], norm: Option<f64>) {
        if 1 < self.dim {
            let mut optimizer = GreedyOptimizer::new(self.dim);
            optimizer.norm = norm;
            optimizer.optim(&xs, &mut self.param);
            let sum: f64 = self.param.iter().sum();
            let scale: f64 = self.param.iter().map(|&p| unsafe { lgamma(p) }).sum();
            self.norm_coef = unsafe { lgamma(sum) - scale };
        }
    }
    pub fn new(param: &[f64]) -> Self {
        let sum: f64 = param.iter().sum();
        let scale: f64 = param.iter().map(|&p| unsafe { lgamma(p) }).sum();
        let norm_coef = unsafe { lgamma(sum) - scale };
        Self {
            dim: param.len(),
            param: param.to_vec(),
            norm_coef,
        }
    }
    pub fn fit(xs: &[f64], norm: Option<f64>) -> Self {
        let dim = xs.len();
        if xs.len() <= 1 || xs.iter().sum::<f64>() < SMALL_VAL {
            let elm = match norm {
                Some(norm) => norm / dim.max(1) as f64,
                None => 1f64,
            };
            let param = vec![elm; dim];
            return Self::new(&param);
        }
        let mut param = match norm {
            Some(norm) => vec![norm * (xs.len() as f64).recip(); xs.len()],
            None => vec![1f64; xs.len()],
        };
        let mut optimizer = GreedyOptimizer::new(xs.len());
        optimizer.norm = norm;
        optimizer.optim(&xs, &mut param);
        if param.iter().any(|x| x.is_nan()) {
            eprintln!("{:?}", xs);
            eprintln!("{:?}", param);
            panic!();
        }
        Self::new(&param)
    }
}

impl std::fmt::Display for Dirichlet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, p) in self.param.iter().enumerate() {
            if i != self.dim - 1 {
                write!(f, "{:.2},", p)?;
            } else {
                write!(f, "{:.2}", p)?;
            }
        }
        write!(f, "]")
    }
}

pub fn fit_data(xs: &[f64]) -> Vec<f64> {
    let dim = xs.len();
    let mut optimizer: GreedyOptimizer = Optimizer::new(dim);
    let mut params: Vec<_> = xs.iter().map(|x| x.exp()).collect();
    let sum: f64 = params.iter().sum();
    params.iter_mut().for_each(|p| *p /= sum);
    optimizer.optim(&xs, &mut params);
    params
}

pub fn fit_data_with<O: Optimizer>(xs: &[f64], optimizer: &mut O, init: &[f64]) -> Vec<f64> {
    let mut params: Vec<_> = init.to_vec();
    optimizer.optim(&xs, &mut params);
    params
}

/// Return the estimated parameters maximixing `Dir(exp(data)|parameters)`
/// Make sure that the input data shoule be logarithmic version of the probability.
/// # Panics
/// Panics if a datum has a non-negative element including zero (i.e., log prob < 0).
/// Panics if a datum does not sum up to 1 when exped.
/// Panics if data do not share the same length.
/// Panics if data is empty.
pub fn fit<D: std::borrow::Borrow<[f64]>>(data: &[D]) -> Vec<f64> {
    let total = data.len() as f64 + SMALL_VAL;
    let mut sums: Vec<_> = data.iter().fold(Vec::new(), |mut acc, xs| {
        if acc.is_empty() {
            acc = xs.borrow().to_vec();
        } else {
            acc.iter_mut()
                .zip(xs.borrow().iter())
                .for_each(|(a, x)| *a += x);
        }
        acc
    });
    if SMALL_VAL < total {
        sums.iter_mut().for_each(|x| *x /= total);
        let dim = sums.len();
        let mut optimizer: GreedyOptimizer = Optimizer::new(dim);
        let mut params: Vec<_> = sums.iter().map(|x| x.exp()).collect();
        let sum: f64 = params.iter().sum();
        params.iter_mut().for_each(|p| *p /= sum);
        optimizer.optim(&sums, &mut params);
        params
    } else {
        vec![1f64; 1]
    }
}

use rand::prelude::SliceRandom;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

/// If the liklihood did not increase `STOP_COUNT` time, it early stops.
pub const STOP_COUNT: usize = 10;
pub trait Optimizer {
    fn new(dim: usize) -> Self;
    fn tik(&mut self);
    fn loop_count(&self) -> usize;
    fn threshold(&self) -> f64;
    fn shuffle(&mut self, data: &mut [(Vec<&[f64]>, (f64, &[f64]))]);
    fn normalize(&self, param: &mut [f64]) {
        if let Some(norm) = self.norm() {
            let sum: f64 = param.iter().map(|x| x * x).sum::<f64>().sqrt();
            param.iter_mut().for_each(|x| *x /= sum);
            param.iter_mut().for_each(|x| *x *= norm)
        }
    }
    /// If there are some L2 constraint on the norm,
    /// return Some(norm), otherwise, none.
    fn norm(&self) -> Option<f64> {
        None
    }
    fn optim(&mut self, data: &[f64], param: &mut [f64]);
}

/// `Norm` is to restrict the parameters to the hyperplane, satisfying the sum of the parameters equal to `norm`.
#[allow(dead_code)]
pub struct GreedyOptimizer {
    dim: usize,
    loop_count: usize,
    // learning rate
    lr: f64,
    // Step size. Usually, 2f64.
    step_size: f64,
    rng: Xoshiro256StarStar,
    threshold: f64,
    norm: Option<f64>,
}

const LEARNING_RATE_BOUND: f64 = 100000f64;
impl GreedyOptimizer {
    pub fn set_norm(mut self, norm: f64) -> Self {
        self.norm = Some(norm);
        self
    }
    pub fn set_threshold(mut self, thr: f64) -> Self {
        self.threshold = thr;
        self
    }
    fn substract_orthogonal_component(grad: &mut [f64]) {
        let sum: f64 = grad.iter().sum();
        let mean = sum / grad.len() as f64;
        grad.iter_mut().for_each(|g| *g -= mean);
    }
    fn doubling_search(&self, grad: &[f64], data: &[f64], param: &[f64]) -> f64 {
        // The 2 is to avoid the parameter from reaching zero.
        let learning_rate_bound: f64 = param
            .iter()
            .zip(grad.iter())
            .filter(|&(_, &g)| g < 0f64)
            .map(|(p, g)| -p / g / 2f64)
            .fold(LEARNING_RATE_BOUND, |x, y| x.min(y));
        let mut lr = self.lr.min(learning_rate_bound);
        let mut jumped_to: Vec<_> = param
            .iter()
            .zip(grad.iter())
            .map(|(p, g)| p + self.lr * g)
            .collect();
        let mut prev = dirichlet_log(data, param);
        let mut current = dirichlet_log(data, &jumped_to);
        assert!(!current.is_nan());
        if prev < current {
            // Increase until the LK begins to decrease.
            while prev < current && lr * self.step_size < learning_rate_bound {
                lr *= self.step_size;
                jumped_to
                    .iter_mut()
                    .zip(param.iter().zip(grad.iter()))
                    .for_each(|(j, (p, g))| *j = p + lr * g);
                // Maximum value so far, .max(prev) is not needed.
                prev = current;
                current = dirichlet_log(data, &jumped_to);
                // trace!("TUNE\t{:.2}\t{}\t{}", current, vec2str(&jumped_to), self.lr);
                assert!(!current.is_nan());
            }
            lr /= self.step_size;
        } else {
            // Decrease learning rate until the likelihood begins to increase.
            // prev is the current maximum. Fix it.
            while current < prev || self.threshold() < prev - current {
                lr /= self.step_size;
                jumped_to
                    .iter_mut()
                    .zip(param.iter().zip(grad.iter()))
                    .for_each(|(j, (p, g))| *j = p + lr * g);
                current = dirichlet_log(data, &jumped_to);
                // trace!("TUNE\t{:.2}\t{}\t{}", current, vec2str(&jumped_to), self.lr);
                assert!(!current.is_nan());
            }
        }
        lr
    }
}

impl Optimizer for GreedyOptimizer {
    fn threshold(&self) -> f64 {
        self.threshold
    }
    fn shuffle(&mut self, data: &mut [(Vec<&[f64]>, (f64, &[f64]))]) {
        data.shuffle(&mut self.rng);
    }
    fn loop_count(&self) -> usize {
        self.loop_count
    }
    fn norm(&self) -> Option<f64> {
        self.norm
    }
    fn tik(&mut self) {
        self.loop_count += 1;
        self.lr /= self.step_size;
    }
    fn new(dim: usize) -> Self {
        Self {
            loop_count: 0,
            dim,
            lr: 0.001,
            step_size: 2f64,
            threshold: 0.000000000001,
            rng: SeedableRng::seed_from_u64(232342),
            norm: None,
        }
    }
    fn optim(&mut self, xs: &[f64], param: &mut [f64]) {
        if let Some(norm) = self.norm() {
            let sum: f64 = param.iter().sum();
            let diff = (sum - norm).abs();
            assert!(
                diff < 0.001,
                "When norm parameter is specified, the initial parameter should sum up to {}({:?})",
                norm,
                param
            );
        }
        let mut current_likelihood = dirichlet_log(&xs, param);
        let before = current_likelihood;
        let mut count_not_increased = 0;
        let mut grad = vec![0f64; param.len()];
        for _ in 0..10000 {
            let param_sum: f64 = param.iter().sum();
            let sum_digam: f64 = digamma(param_sum);
            grad.iter_mut()
                .zip(param.iter())
                .zip(xs.iter())
                .for_each(|((g, &p), &x)| *g = sum_digam - digamma(p) + x);
            if self.norm.is_some() {
                Self::substract_orthogonal_component(&mut grad);
            }
            self.lr = self.doubling_search(&grad, xs, param);
            param
                .iter_mut()
                .zip(grad.iter())
                .for_each(|(p, g)| *p += self.lr * g);
            let likelihood = dirichlet_log(&xs, param);
            self.tik();
            if likelihood <= current_likelihood + self.threshold() {
                count_not_increased += 1;
            } else {
                count_not_increased = 0;
            }
            current_likelihood = current_likelihood.max(likelihood);
            if STOP_COUNT < count_not_increased {
                break;
            }
        }
        if let Some(norm) = self.norm() {
            let sum: f64 = param.iter().sum();
            assert!((sum - norm).abs() < 0.001, "{}({:?})", norm, param);
        }
        let after = current_likelihood;
        assert!(before <= after);
    }
}

#[allow(dead_code)]
fn vec2str(xs: &[f64]) -> String {
    let xs: Vec<_> = xs.iter().map(|x| format!("{:.2}", x)).collect();
    xs.join(",")
}

// fn get_gradient(ds: &[&[f64]], ws: &[f64], param: &[f64]) -> Vec<f64> {
//     let param_sum: f64 = param.iter().sum();
//     let sum_digam: f64 = digamma(param_sum);
//     let weight_sum: f64 = ws.iter().sum::<f64>();
//     let mut grad: Vec<_> = param
//         .iter()
//         .map(|&p| weight_sum * (sum_digam - digamma(p)))
//         .collect();
//     for (w, logprob) in ws.iter().zip(ds.iter()) {
//         for (grad, x) in grad.iter_mut().zip(logprob.iter()) {
//             *grad += w * x;
//         }
//     }
//     grad
// }

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

// Return the component of the `grad` which is orthogonal to the `param`.
// fn to_orthogonal(mut grad: Vec<f64>, param: &[f64]) -> Vec<f64> {
//     let norm: f64 = param.iter().map(|x| x * x).sum();
//     let inner_product: f64 = grad.iter().zip(param.iter()).map(|(x, y)| x * y).sum();
//     grad.iter_mut()
//         .zip(param.iter())
//         .for_each(|(g, p)| *g -= inner_product * p / norm);
//     grad
// }

// /// Adam Optimizer.
// pub struct AdamOptimizer {
//     dim: usize,
//     loop_count: usize,
//     lr: f64,
//     beta_1: f64,
//     beta_2: f64,
//     epsilon: f64,
//     decay: f64,
//     moment: Vec<f64>,
//     sq_moment: Vec<f64>,
//     rng: Xoshiro256StarStar,
//     threshold: f64,
//     norm: Option<f64>,
// }

// impl AdamOptimizer {
//     /// # Arguments
//     /// - `dim`: Dimension of the Dirichlet distribution to be estimated.
//     /// - `learning_rate`: Learning rate of the Adam. Usually near 0, Recommend 0.01~0.03.
//     /// - `beta_1`: Decay rate of the 1st order moment. Recommend 0.9.
//     /// - `beta_2`: Decay rate of the 2nd order moment. Recommend 0.999.
//     /// - `decay`: decay rate of the learinng rate. Recommend 1.
//     /// - `seed`: seed used in the random number generator in AdamOptimizer.
//     /// - `threshold`: if the lielihood did not increased `threshold` for T(=10) times, it early drops.
//     ///
//     pub fn with_details(
//         dim: usize,
//         learning_rate: f64,
//         (beta_1, beta_2): (f64, f64),
//         decay: f64,
//         threshold: f64,
//         norm: Option<f64>,
//         seed: u64,
//     ) -> Self {
//         Self {
//             dim,
//             loop_count: 0,
//             lr: learning_rate,
//             threshold,
//             beta_1,
//             beta_2,
//             epsilon: 0.00000001,
//             decay,
//             moment: vec![0f64; dim],
//             sq_moment: vec![0f64; dim],
//             rng: SeedableRng::seed_from_u64(seed),
//             norm,
//         }
//     }
//     /// Change learning rate.
//     pub fn learning_rate(mut self, lr: f64) -> Self {
//         self.lr = lr;
//         self
//     }
//     /// Set normalize constraint.
//     pub fn norm(mut self, norm: f64) -> Self {
//         self.norm = Some(norm);
//         self
//     }
// }

// impl Optimizer for AdamOptimizer {
//     fn threshold(&self) -> f64 {
//         self.threshold
//     }
//     fn norm(&self) -> Option<f64> {
//         self.norm
//     }
//     fn new(dim: usize) -> Self {
//         Self {
//             dim,
//             loop_count: 0,
//             lr: 0.01,
//             beta_1: 0.9,
//             beta_2: 0.999,
//             epsilon: 0.0000001,
//             decay: 0.995,
//             threshold: 0.0001,
//             moment: vec![0f64; dim],
//             sq_moment: vec![0f64; dim],
//             rng: SeedableRng::seed_from_u64(34820),
//             norm: None,
//         }
//     }
//     fn tik(&mut self) {
//         self.lr *= self.decay;
//         self.loop_count += 1;
//     }
//     fn update(&mut self, ds: &[&[f64]], (w_data, ws): (f64, &[f64]), param: &mut [f64]) {
//         // Calculate gradient.
//         // let grad = get_gradient(ds, w_data, ws, param);
//         let mut grad = get_gradient(ds, ws, param);
//         grad.iter_mut().for_each(|g| *g *= w_data);
//         // If normalization is on, calc orthogonal vector.
//         let grad = match self.norm {
//             Some(_) => to_orthogonal(grad, param),
//             None => grad,
//         };
//         // Update moment, sq_moment.
//         self.moment
//             .iter_mut()
//             .zip(grad.iter())
//             .for_each(|(m, g)| *m = self.beta_1 * *m + (1.0 - self.beta_1) * g);
//         self.sq_moment
//             .iter_mut()
//             .zip(grad.iter())
//             .for_each(|(v, g)| *v = self.beta_2 * *v + (1.0 - self.beta_2) * g * g);
//         // calc gradient.
//         let factor_1 = 1f64 - self.beta_1.powi(self.loop_count as i32 + 1);
//         let factor_2 = 1f64 - self.beta_2.powi(self.loop_count as i32 + 1);
//         let grad: Vec<_> = self.moment.iter().map(|m| m / factor_1).collect();
//         let var: Vec<_> = self.sq_moment.iter().map(|v| v / factor_2).collect();
//         let grad: Vec<_> = grad
//             .into_iter()
//             .zip(var.iter())
//             .map(|(g, v)| {
//                 let div = v.sqrt() + self.epsilon;
//                 self.lr * g / div
//             })
//             .collect();
//         // Scaling...
//         let scaler = grad
//             .iter()
//             .zip(param.iter())
//             .filter(|&(g, p)| p + g < 0f64)
//             .map(|(g, p)| 2.0 * g.abs() / p)
//             .max_by(|x, y| x.partial_cmp(y).unwrap())
//             .unwrap_or(1f64);
//         // trace!(
//         //     "GRAD\t{}\t{}\t{}\t{}",
//         //     self.loop_count(),
//         //     vec2str(&param),
//         //     vec2str(&grad),
//         //     scaler,
//         // );
//         param
//             .iter_mut()
//             .zip(grad)
//             .for_each(|(p, g)| *p += g / scaler);
//     }
//     fn loop_count(&self) -> usize {
//         self.loop_count
//     }
//     fn shuffle(&mut self, data: &mut [(Vec<&[f64]>, (f64, &[f64]))]) {
//         data.shuffle(&mut self.rng);
//     }
// }

// #[allow(dead_code)]
// pub struct MomentumOptimizer {
//     dim: usize,
//     loop_count: usize,
//     lr: f64,
//     // Decay rate of the momentum
//     decay: f64,
//     moment: Vec<f64>,
//     rng: Xoshiro256StarStar,
//     threshold: f64,
// }

// impl Optimizer for MomentumOptimizer {
//     fn threshold(&self) -> f64 {
//         self.threshold
//     }
//     fn shuffle(&mut self, data: &mut [(Vec<&[f64]>, (f64, &[f64]))]) {
//         data.shuffle(&mut self.rng);
//     }
//     fn new(dim: usize) -> Self {
//         Self {
//             dim,
//             loop_count: 0,
//             lr: 0.01,
//             decay: 0.9,
//             threshold: 0.0001,
//             moment: vec![0f64; dim],
//             rng: SeedableRng::seed_from_u64(432543543),
//         }
//     }
//     fn tik(&mut self) {
//         self.loop_count += 1;
//     }
//     fn update(&mut self, ds: &[&[f64]], (w_data, ws): (f64, &[f64]), param: &mut [f64]) {
//         // Calc crad
//         let mut grad = get_gradient(ds, ws, param);
//         grad.iter_mut().for_each(|g| *g *= w_data);
//         // Merge with current moment.
//         let lr = self.lr * (1f64 - self.decay);
//         self.moment
//             .iter_mut()
//             .zip(grad)
//             .for_each(|(m, g)| *m = self.decay * *m + lr * g);
//         // Reduce if the moment is too large.
//         if let Some(scale) = self
//             .moment
//             .iter()
//             .zip(param.iter())
//             .filter(|&(g, p)| p + g < 0f64)
//             .map(|(g, p)| p / 2.0 / g)
//             .min_by(|x, y| x.partial_cmp(y).unwrap())
//         {
//             self.moment.iter_mut().for_each(|x| *x /= scale);
//         }
//         // let scalar: f64 = self.moment.iter().map(|x| x * x).sum();
//         // trace!(
//         //     "GRAD\t{}\t{:.2}\t[{}]",
//         //     self.loop_count(),
//         //     scalar,
//         //     vec2str(&self.moment)
//         // );
//         param
//             .iter_mut()
//             .zip(&self.moment)
//             .for_each(|(p, m)| *p += m);
//     }
//     fn loop_count(&self) -> usize {
//         self.loop_count
//     }
// }

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
