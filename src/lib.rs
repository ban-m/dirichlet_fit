#[allow(unused_imports)]
#[macro_use]
extern crate log;
const SMALL_VAL: f64 = 0.00000001;
const LOG_FILTER: f64 = -30f64;
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
    pub fn param(&self) -> &[f64] {
        &self.param
    }
    pub fn update(&mut self, xs: &[f64], norm: Option<f64>) {
        if 1 < self.dim {
            let prev = self.clone();
            let lk = self.lk(xs);
            let mut buf1 = vec![0f64; xs.len()];
            match norm {
                Some(norm) => {
                    let sum: f64 = self.param.iter().sum();
                    self.param.iter_mut().for_each(|p| *p = (*p * norm) / sum);
                    fit_dirichlet_only_mean(xs, &mut self.param, &mut buf1)
                }
                None => fit_dirichlet_with(xs, &mut self.param, &mut buf1),
            }
            let sum: f64 = self.param.iter().sum();
            let scale: f64 = self.param.iter().map(|&p| unsafe { lgamma(p) }).sum();
            self.norm_coef = unsafe { lgamma(sum) - scale };
            let new_lk = self.lk(xs);
            if new_lk < lk {
                *self = prev;
            }
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
    /// Input:Loged version of the observation
    pub fn fit(xs: &[f64], norm: Option<f64>) -> Self {
        let dim = xs.len();
        if xs.len() <= 1 {
            let elm = match norm {
                Some(norm) => norm / dim.max(1) as f64,
                None => 1f64,
            };
            let param = vec![elm; dim];
            return Self::new(&param);
        }
        let mut param = estim_initial_param(xs);
        let mut buf1 = vec![0f64; xs.len()];
        if let Some(norm) = norm {
            param.iter_mut().for_each(|x| *x *= norm);
            fit_dirichlet_only_mean(xs, &mut param, &mut buf1)
        } else {
            fit_dirichlet_with(xs, &mut param, &mut buf1)
        }
        Self::new(&param)
    }
}

// xs:log probs
fn estim_initial_param(xs: &[f64]) -> Vec<f64> {
    let mut param: Vec<_> = xs.iter().map(|&x| digam_inv(x)).collect();
    let sum: f64 = param.iter().sum();
    param.iter_mut().for_each(|x| *x /= sum);
    param
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

// pub fn fit_data(xs: &[f64]) -> Vec<f64> {
//     let dim = xs.len();
//     let mut optimizer: GreedyOptimizer = Optimizer::new(dim);
//     let mut params: Vec<_> = xs.iter().map(|x| x.exp()).collect();
//     let sum: f64 = params.iter().sum();
//     params.iter_mut().for_each(|p| *p /= sum);
//     optimizer.optim(&xs, &mut params);
//     params
// }

// pub fn fit_data_with<O: Optimizer>(xs: &[f64], optimizer: &mut O, init: &[f64]) -> Vec<f64> {
//     let mut params: Vec<_> = init.to_vec();
//     optimizer.optim(&xs, &mut params);
//     params
// }

// /// Return the estimated parameters maximixing `Dir(exp(data)|parameters)`
// /// Make sure that the input data shoule be logarithmic version of the probability.
// /// # Panics
// /// Panics if a datum has a non-negative element including zero (i.e., log prob < 0).
// /// Panics if a datum does not sum up to 1 when exped.
// /// Panics if data do not share the same length.
// /// Panics if data is empty.
// pub fn fit<D: std::borrow::Borrow<[f64]>>(data: &[D]) -> Vec<f64> {
//     let total = data.len() as f64 + SMALL_VAL;
//     let mut sums: Vec<_> = data.iter().fold(Vec::new(), |mut acc, xs| {
//         if acc.is_empty() {
//             acc = xs.borrow().to_vec();
//         } else {
//             acc.iter_mut()
//                 .zip(xs.borrow().iter())
//                 .for_each(|(a, x)| *a += x);
//         }
//         acc
//     });
//     if SMALL_VAL < total {
//         sums.iter_mut().for_each(|x| *x /= total);
//         let dim = sums.len();
//         let mut optimizer: GreedyOptimizer = Optimizer::new(dim);
//         let mut params: Vec<_> = sums.iter().map(|x| x.exp()).collect();
//         let sum: f64 = params.iter().sum();
//         params.iter_mut().for_each(|p| *p /= sum);
//         optimizer.optim(&sums, &mut params);
//         params
//     } else {
//         vec![1f64; 1]
//     }
// }

// // use rand::SeedableRng;
// // use rand_xoshiro::Xoshiro256StarStar;

// /// If the liklihood did not increase `STOP_COUNT` time, it early stops.
// pub const STOP_COUNT: usize = 10;
// pub trait Optimizer {
//     fn new(dim: usize) -> Self;
//     fn tik(&mut self);
//     fn loop_count(&self) -> usize;
//     fn threshold(&self) -> f64;
//     fn norm(&self) -> Option<f64>;
//     fn optim(&mut self, data: &[f64], param: &mut [f64]);
// }

// /// `Norm` is to restrict the parameters to the hyperplane, satisfying the sum of the parameters equal to `norm`.
// #[allow(dead_code)]
// pub struct GreedyOptimizer {
//     dim: usize,
//     loop_count: usize,
//     // learning rate
//     lr: f64,
//     step_size: f64,
//     // rng: Xoshiro256StarStar,
//     threshold: f64,
//     norm: Option<f64>,
//     // If the norm is on and the value is less than `max` * `mas_ratio`, then,
//     // the corresponding parameter would nobe fixed to very small value.
//     // mask_ratio: Option<f64>,
// }

// const LEARNING_RATE_BOUND: f64 = 100000f64;
// impl GreedyOptimizer {
//     pub fn set_norm(mut self, norm: f64) -> Self {
//         self.norm = Some(norm);
//         self
//     }
//     pub fn set_threshold(mut self, thr: f64) -> Self {
//         self.threshold = thr;
//         self
//     }
//     // pub fn set_mask_ratio(mut self, ratio: f64) -> Self {
//     //     self.mask_ratio = Some(ratio);
//     //     self
//     // }
//     // pub fn with(norm: f64, ratio: f64, dim: usize) -> Self {
//     //     Self {
//     //         loop_count: 0,
//     //         dim,
//     //         lr: 0.01,
//     //         step_size: 1.5,
//     //         threshold: 0.000000000001,
//     //         norm: Some(norm),
//     //         mask_ratio: Some(ratio),
//     //     }
//     // }
//     // fn substract_orthogonal_component(grad: &mut [f64]) {
//     //     let sum: f64 = grad.iter().sum();
//     //     let mean = sum / grad.len() as f64;
//     //     grad.iter_mut().for_each(|g| *g -= mean);
//     // }
//     fn doubling_search(&self, grad: &[f64], data: &[f64], param: &[f64]) -> f64 {
//         // The 2 is to avoid the parameter from reaching zero.
//         let learning_rate_bound: f64 = param
//             .iter()
//             .zip(grad.iter())
//             .filter(|&(_, &g)| g < 0f64)
//             .fold(LEARNING_RATE_BOUND, |x, (p, g)| x.min(-p / g / 2f64));
//         let mut lr = self.lr.min(learning_rate_bound);
//         let jumped_to = param.iter().zip(grad.iter()).map(|(p, g)| p + lr * g);
//         let mut prev = dirichlet_log(data, param);
//         let mut current = dirichlet_log_str(data, jumped_to);
//         if prev < current {
//             while prev < current && lr * self.step_size < learning_rate_bound {
//                 lr *= self.step_size;
//                 let jumped_to = param.iter().zip(grad.iter()).map(|(p, g)| p + lr * g);
//                 prev = current;
//                 current = dirichlet_log_str(data, jumped_to);
//             }
//             lr / self.step_size
//         } else {
//             while current < prev || self.threshold() < prev - current {
//                 lr /= self.step_size;
//                 let jumped_to = param.iter().zip(grad.iter()).map(|(p, g)| p + lr * g);
//                 prev = current;
//                 current = dirichlet_log_str(data, jumped_to);
//             }
//             lr
//         }
//     }
// }

// impl Optimizer for GreedyOptimizer {
//     fn threshold(&self) -> f64 {
//         self.threshold
//     }
//     // fn shuffle(&mut self, data: &mut [(Vec<&[f64]>, (f64, &[f64]))]) {
//     //     data.shuffle(&mut self.rng);
//     // }
//     fn loop_count(&self) -> usize {
//         self.loop_count
//     }
//     fn norm(&self) -> Option<f64> {
//         self.norm
//     }
//     fn tik(&mut self) {
//         self.loop_count += 1;
//         self.lr /= self.step_size;
//     }
//     fn new(dim: usize) -> Self {
//         Self {
//             loop_count: 0,
//             dim,
//             lr: 0.04,
//             step_size: 1.5,
//             threshold: SMALL_VAL,
//             // rng: SeedableRng::seed_from_u64(232342),
//             norm: None,
//             // mask_ratio: None,
//         }
//     }
//     fn optim(&mut self, xs: &[f64], param: &mut [f64]) {
//         if let Some(norm) = self.norm() {
//             let sum: f64 = param.iter().sum();
//             let diff = (sum - norm).abs();
//             assert!(
//                 diff < 0.001,
//                 "When norm parameter is specified, the initial parameter should sum up to {}({:?})",
//                 norm,
//                 param
//             );
//         }
//         let init = param.to_vec();
//         let before = dirichlet_log(&xs, param);
//         // let prev_param = param.to_vec();
//         let mut grad: Vec<_> = vec![0f64; param.len()];
//         // for t in 1..1000 {
//         //     let param_sum: f64 = param.iter().sum();
//         //     let sum_digam: f64 = digamma(param_sum);
//         //     grad.iter_mut()
//         //         .zip(param.iter())
//         //         .zip(xs.iter())
//         //         .for_each(|((g, &p), &x)| *g = sum_digam - digamma(p) + x);
//         //     let step_size_bound: f64 = param
//         //         .iter()
//         //         .zip(grad.iter())
//         //         .filter(|&(_, &g)| g < 0f64)
//         //         .fold(LEARNING_RATE_BOUND, |x, (p, g)| x.min(-p / g / 2f64));
//         //     let step_size = (self.lr / (t as f64).sqrt()).min(step_size_bound);
//         //     param
//         //         .iter_mut()
//         //         .zip(grad.iter())
//         //         .for_each(|(p, g)| *p += step_size * g);
//         //     debug!("BF\t{}\t{}\t{}", vec2str(&param), vec2str(&grad), step_size);
//         //     if let Some(norm) = self.norm {
//         //         projection(param, &mut grad, norm);
//         //         param.iter_mut().for_each(|x| *x = x.max(SMALL_VAL));
//         //         assert!(param.iter().all(|x| x.is_sign_positive()));
//         //         let diff = (param.iter().sum::<f64>() - norm).abs();
//         //         assert!(diff < 0.00001, "{:?}\t{}", param, diff);
//         //     }
//         //     let likelihood = dirichlet_log(&xs, param);
//         //     debug!("AF\t{}\t{:.4}", vec2str(&param), likelihood);
//         //     let diff: f64 = param
//         //         .iter()
//         //         .zip(prev_param.iter())
//         //         .map(|(x, y)| (x - y) * (x - y))
//         //         .sum();
//         //     if diff < self.threshold() {
//         //         break;
//         //     } else {
//         //         prev_param.clone_from_slice(param);
//         //     }
//         // }
//         let mut current_likelihood = dirichlet_log(&xs, param);
//         let mut count_not_increased = 0;
//         for _ in 1..10000 {
//             let param_sum: f64 = param.iter().sum();
//             let sum_digam: f64 = digamma(param_sum);
//             grad.iter_mut()
//                 .zip(param.iter())
//                 .zip(xs.iter())
//                 .for_each(|((g, &p), &x)| *g = sum_digam - digamma(p) + x);
//             // if self.norm.is_some() {
//             //     Self::substract_orthogonal_component(&mut grad);
//             // }
//             self.lr = self.doubling_search(&grad, xs, param);
//             param
//                 .iter_mut()
//                 .zip(grad.iter())
//                 .for_each(|(p, g)| *p += self.lr * g);
//             if let Some(norm) = self.norm {
//                 let sum: f64 = param.iter().sum();
//                 param.iter_mut().for_each(|p| *p = *p * norm / sum);
//             }
//             let likelihood = dirichlet_log(&xs, param);
//             debug!("{}\t{}\t{:.4}", vec2str(&grad), self.lr.log10(), likelihood);
//             self.tik();
//             if likelihood <= current_likelihood + self.threshold() {
//                 count_not_increased += 1;
//             } else {
//                 count_not_increased = 0;
//             }
//             current_likelihood = current_likelihood.max(likelihood);
//             if STOP_COUNT < count_not_increased {
//                 break;
//             }
//         }
//         if let Some(norm) = self.norm() {
//             let sum: f64 = param.iter().sum();
//             assert!((sum - norm).abs() < 0.001, "{}({:?})", norm, param);
//         }
//         let after = dirichlet_log(&xs, param);
//         if after < before {
//             param.clone_from_slice(&init);
//         }
//     }
// }

// fn dirichlet_log_str<I: std::iter::Iterator<Item = f64>>(log_prob: &[f64], parameters: I) -> f64 {
//     let mut sum = 0f64;
//     // Scaling factor.
//     let mut scale = 0f64;
//     // Likelihood depends on `probability`
//     let mut lk = 0f64;
//     for (param, log_prob) in parameters.zip(log_prob.iter()) {
//         sum += param;
//         scale += unsafe { lgamma(param) };
//         lk += (param - 1f64) * log_prob;
//     }
//     unsafe { lgamma(sum) - scale + lk }
// }

// fn projection(xs: &mut [f64], buffer: &mut [f64], norm: f64) {
//     if xs.is_empty() {
//         return;
//     } else if xs.len() <= 1 {
//         xs[0] = norm
//     } else {
//         buffer.copy_from_slice(xs);
//         buffer.sort_by(|x, y| y.partial_cmp(x).unwrap());
//         let mut part_sum = 0f64;
//         for i in 0..buffer.len() - 1 {
//             part_sum += buffer[i];
//             let part_average = (part_sum - norm) / (i + 1) as f64;
//             if buffer[i + 1] < part_average {
//                 let offset = part_average;
//                 xs.iter_mut().for_each(|x| *x = (*x - offset).max(0f64));
//                 return;
//             }
//         }
//         part_sum += buffer.last().unwrap();
//         let offset = (part_sum - norm) / buffer.len() as f64;
//         xs.iter_mut().for_each(|x| *x = (*x - offset).max(0f64));
//     }
// }

#[allow(dead_code)]
fn vec2str(xs: &[f64]) -> String {
    let xs: Vec<_> = xs.iter().map(|x| format!("{:6.2}", x)).collect();
    xs.join("\t")
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

fn trigamma(x: f64) -> f64 {
    if x < 10f64 {
        let offset: f64 = (0..10)
            .map(|i| {
                let x = i as f64 + x;
                (x * x).recip()
            })
            .sum();
        trigamma_large(x + 10f64) + offset
    } else {
        trigamma_large(x)
    }
}

fn trigamma_large(x: f64) -> f64 {
    let mut trigam_val = x.recip();
    trigam_val += (2.0 * x.powi(2)).recip();
    trigam_val += (6.0 * x.powi(3)).recip();
    trigam_val -= (30.0 * x.powi(5)).recip();
    trigam_val += (42.0 * x.powi(7)).recip();
    trigam_val -= (30.0 * x.powi(9)).recip();
    trigam_val += 5.0 / (66.0 * x.powi(11));
    trigam_val -= 691.0 / (2730.0 * x.powi(13));
    trigam_val += 7.0 / (6.0 * x.powi(15));
    trigam_val
}

pub fn fit_dirichlet(xs: &[f64]) -> Vec<f64> {
    let mut param: Vec<_> = estim_initial_param(xs);
    let mut buf1 = vec![0f64; param.len()];
    fit_dirichlet_with(xs, &mut param, &mut buf1);
    param
}

/// Fit only mean, not precision.
pub fn fit_dirichlet_only_mean(xs: &[f64], param: &mut [f64], buf1: &mut [f64]) {
    let precision: f64 = param.iter().sum();
    param.iter_mut().for_each(|x| *x /= precision);
    update_new_mean(precision, xs, param, buf1);
    param.iter_mut().for_each(|x| *x *= precision);
}

pub fn fit_dirichlet_with(xs: &[f64], param: &mut [f64], buf1: &mut [f64]) {
    // Mask xs with very small value, less than LOG_FILTER
    loop {
        buf1.clone_from_slice(param);
        let sum: f64 = param.iter().sum();
        param
            .iter_mut()
            .zip(xs.iter())
            .filter(|&(_, &x)| LOG_FILTER < x)
            .for_each(|(p, x)| {
                *p = digam_inv(digamma(sum) + x);
            });
        let diff: f64 = buf1
            .iter()
            .zip(param.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum();
        if diff < SMALL_VAL {
            break;
        }
    }
}

fn update_new_mean(prec: f64, xs: &[f64], param: &mut [f64], buf: &mut [f64]) {
    let mut prev_diff = 10000000000000000000000000000000f64;
    let retain_sum: f64 = param
        .iter()
        .zip(xs.iter())
        .filter(|&(_, &x)| x < LOG_FILTER)
        .fold(0f64, |x, (p, _)| x + p);
    let distrib = 1f64 - retain_sum;
    for _t in 0.. {
        buf.clone_from_slice(param);
        let offset: f64 = param
            .iter()
            .zip(xs.iter())
            .filter(|&(_, &x)| LOG_FILTER < x)
            .map(|(p, x)| p * (digamma(prec * p) - x))
            .sum();
        param
            .iter_mut()
            .zip(xs.iter())
            .filter(|&(_, &x)| LOG_FILTER < x)
            .for_each(|(p, x)| *p = digam_inv_safe(x + offset));
        let sum: f64 = param
            .iter()
            .zip(xs.iter())
            .filter(|&(_, &x)| LOG_FILTER < x)
            .fold(0f64, |x, (y, _)| x + y);
        let factor = sum / distrib;
        param
            .iter_mut()
            .zip(xs.iter())
            .filter(|&(_, &x)| LOG_FILTER < x)
            .for_each(|(p, _)| *p /= factor);
        let sum: f64 = param.iter().sum();
        assert!((1f64 - sum).abs() < 0.0001, "{:?}", param);
        let diff: f64 = param
            .iter()
            .zip(buf.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum();
        if diff < SMALL_VAL {
            break;
        }
        if (diff - prev_diff).abs() < SMALL_VAL {
            stop_after_greedy(prec, xs, param, buf);
            break;
        }
        prev_diff = diff;
    }
}

fn stop_after_greedy(prec: f64, xs: &[f64], param: &mut [f64], buf: &mut [f64]) {
    trace!("FALLBACK\t{:?}", xs);
    let mut loss: f64 = xs
        .iter()
        .zip(param.iter())
        .map(|(x, p)| prec * p * x - unsafe { lgamma(prec * p) })
        .sum();
    loop {
        buf.clone_from_slice(param);
        let offset: f64 = param
            .iter()
            .zip(xs.iter())
            .map(|(p, x)| p * (digamma(prec * p) - x))
            .sum();
        param
            .iter_mut()
            .zip(xs.iter())
            .for_each(|(p, x)| *p = digam_inv(x + offset));
        let sum: f64 = param.iter().sum();
        param.iter_mut().for_each(|x| *x /= sum);
        let new_loss: f64 = xs
            .iter()
            .zip(param.iter())
            .map(|(x, p)| prec * p * x - unsafe { lgamma(prec * p) })
            .sum();
        if loss < new_loss {
            // recover and break.
            param.clone_from_slice(buf);
            break;
        }
        loss = new_loss;
    }
}

// Get new guess of the precision by fix-point iteration.
// fn get_new_precision(prec: f64, xs: &[f64], param: &[f64]) -> f64 {
//     let mut guess = prec;
//     let offset: f64 = param.iter().zip(xs.iter()).map(|(p, x)| p * x).sum();
//     loop {
//         let before = guess;
//         let grad =
//             digamma(guess) - param.iter().map(|p| p * digamma(guess * p)).sum::<f64>() + offset;
//         let curv = trigamma(guess)
//             - param
//                 .iter()
//                 .map(|p| p * p * trigamma(guess * p))
//                 .sum::<f64>();
//         guess = (guess.recip() + (guess * guess).recip() * grad / curv).recip();
//         if (before - guess).powi(2) < SMALL_VAL {
//             break;
//         }
//     }
//     guess
// }

// This is *safer* version of the digamma inverse function,
// namely, if the y is greater than 20,
// we consider digam(x) ~ ln x - 1/2x < 20 and
// usualy x is very very large, thus 1/2x can be ignored.
// In other words, the inverse function is just exp(y).
const THRESHOLD: f64 = 20f64;
const LOOP_THR: usize = 1000;
fn digam_inv_safe(y: f64) -> f64 {
    if THRESHOLD < y {
        return y.exp();
    }
    let mut guess = if -2.22 < y {
        y.exp() + 0.5
    } else {
        -(y - DIGAM_ONE).recip()
    };
    for t in 0.. {
        let prev = guess;
        guess -= (digamma(guess) - y) / trigamma(guess);
        if (prev - guess).powi(2) < SMALL_VAL {
            break;
        }
        if LOOP_THR < t {
            return y.exp();
        }
    }
    guess
}

/// digamma(1)
pub const DIGAM_ONE: f64 = -0.5772156649015328606065;
// Return inverse of digamma function.
pub fn digam_inv(y: f64) -> f64 {
    let mut guess = if -2.22 < y {
        y.exp() + 0.5
    } else {
        -(y - DIGAM_ONE).recip()
    };
    for t in 0.. {
        let prev = guess;
        guess -= (digamma(guess) - y) / trigamma(guess);
        if (prev - guess).powi(2) < SMALL_VAL {
            break;
        }
        if t > 1_000_000 {
            panic!("{}", y);
        }
    }
    guess
}

#[cfg(test)]
pub mod tests {
    use super::*;
    #[test]
    fn trigamma_test() {
        let trig = trigamma(3.0);
        let answer = 0.3949340;
        let diff = (trig - answer).abs();
        assert!(diff < 0.001, "{},{}", diff, trig);
        let trig = trigamma(33.0);
        let answer = 0.03076680;
        let diff = (trig - answer).abs();
        assert!(diff < 0.001, "{},{}", diff, trig);
        let trig = trigamma(0.1);
        let answer = 101.433299;
        let diff = (trig - answer).abs();
        assert!(diff < 0.001, "{},{}", diff, trig)
    }
    #[test]
    fn it_works() {}
    // #[test]
    // fn projection_test() {
    //     let mut buffer = vec![0f64; 2];
    //     let mut xs = vec![6f64, 1f64];
    //     projection(&mut xs, &mut buffer, 1f64);
    //     let answer = vec![1f64, 0f64];
    //     let diff: f64 = xs
    //         .iter()
    //         .zip(answer.iter())
    //         .map(|(x, y)| (x - y).powi(2))
    //         .sum();
    //     assert!(diff < 0.00001, "{:?},{:?}", xs, answer);
    //     let mut xs = vec![6f64, 1f64];
    //     projection(&mut xs, &mut buffer, 2f64);
    //     let answer = vec![2f64, 0f64];
    //     let diff: f64 = xs
    //         .iter()
    //         .zip(answer.iter())
    //         .map(|(x, y)| (x - y).powi(2))
    //         .sum();
    //     assert!(diff < 0.00001, "{:?},{:?}", xs, answer);
    //     let mut xs = vec![5f64, 5f64];
    //     projection(&mut xs, &mut buffer, 2f64);
    //     let answer = vec![1f64, 1f64];
    //     let diff: f64 = xs
    //         .iter()
    //         .zip(answer.iter())
    //         .map(|(x, y)| (x - y).powi(2))
    //         .sum();
    //     assert!(diff < 0.00001, "{:?},{:?}", xs, answer);
    //     let mut xs = vec![100f64, 1f64, 1f64];
    //     let mut buffer = vec![0f64; 3];
    //     projection(&mut xs, &mut buffer, 2f64);
    //     let answer = vec![2f64, 0f64, 0f64];
    //     let diff: f64 = xs
    //         .iter()
    //         .zip(answer.iter())
    //         .map(|(x, y)| (x - y).powi(2))
    //         .sum();
    //     assert!(diff < 0.00001, "{:?},{:?}", xs, answer);
    // }
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
    fn digam_one() {
        let dig = digamma(1.0);
        assert!((dig - DIGAM_ONE).abs() < 0.0001, "{},{}", dig, DIGAM_ONE);
    }
    #[test]
    fn digam_inv_test() {
        for x in [0.01, 0.1, 1.0, 2.0, 10.0, 100.0] {
            let dig = digamma(x);
            let inved = digam_inv(dig);
            assert!((x - inved).abs() < 0.001, "{},{},{}", x, dig, inved);
        }
    }
    #[test]
    fn optim_test_single() {}
}
