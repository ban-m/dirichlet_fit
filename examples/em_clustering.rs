#[macro_use]
extern crate log;
use dirichlet_fit::Optimizer;
use rand::{Rng, SeedableRng};
use rand_distr::{self, Distribution};
use rand_xoshiro::Xoshiro256PlusPlus;

fn gen_post_dist<R: Rng>(cl: usize, dim: usize, err: bool, rng: &mut R) -> Vec<f64> {
    let mut param = vec![0.5f64; dim];
    if !err {
        param[cl] = 20.0;
    }
    let dirichlet = rand_distr::Dirichlet::new(&param).unwrap();
    dirichlet.sample(rng).into_iter().map(|x| x.ln()).collect()
}

fn main() {
    env_logger::init();
    let mut rng: Xoshiro256PlusPlus = SeedableRng::seed_from_u64(4234);
    let len = 10;
    let centers: Vec<_> = (0..2 * len)
        .map(|i| gen_post_dist(i / len, 2, false, &mut rng))
        .collect();
    center_clustering(&centers, 2, &mut rng);
}
fn sum_and_normalize(xss: &[Vec<f64>]) -> Vec<f64> {
    let mut sumed = vec![0f64; xss[0].len()];
    let mut total = 0f64;
    for xs in xss {
        sumed
            .iter_mut()
            .zip(xs.iter())
            .for_each(|(acc, x)| *acc += x);
        total += xs.iter().sum::<f64>();
    }
    sumed.iter_mut().for_each(|x| *x /= total);
    sumed
}

fn center_clustering<R: Rng>(centers: &[Vec<f64>], k: usize, rng: &mut R) {
    let dir = rand_distr::Dirichlet::new(&vec![0.5f64; k]).unwrap();
    let mut weights: Vec<_> = centers.iter().map(|_| dir.sample(rng)).collect();
    // let mut weights = vec![
    //     vec![vec![0.01, 0.99]; centers.len() / 2],
    //     vec![vec![0.99, 0.01]; centers.len() / 2],
    // ]
    // .concat();
    let mut params: Vec<_> = (0..k)
        .map(|cl| {
            let weights: Vec<_> = weights.iter().map(|ws| ws[cl]).collect();
            dirichlet_fit::fit_multiple(&[centers], &[(1f64, weights)])
        })
        .collect();
    let mut fractions = sum_and_normalize(&weights);
    fn lk(centers: &[Vec<f64>], params: &[Vec<f64>], fractions: &[f64]) -> f64 {
        centers
            .iter()
            .map(|probs| {
                let lks: Vec<_> = params
                    .iter()
                    .zip(fractions.iter())
                    .map(|(p, f)| f.ln() + dirichlet_fit::dirichlet_log(probs, p))
                    .collect();
                logsumexp(&lks)
            })
            .sum()
    }
    let mut current_lk = lk(centers, &params, &fractions);
    trace!("LK\t{}\t{}", 1, current_lk);
    for t in 1..100 {
        // Update weight.
        for (center, weight) in centers.iter().zip(weights.iter_mut()) {
            let lks: Vec<_> = params
                .iter()
                .zip(fractions.iter())
                .map(|(p, f)| f.ln() + dirichlet_fit::dirichlet_log(center, p))
                .collect();
            let total = logsumexp(&lks);
            for (w, lk) in weight.iter_mut().zip(lks) {
                *w = (lk - total).exp();
            }
        }
        // Update parameters.
        fractions = sum_and_normalize(&weights);
        for (cl, param) in params.iter_mut().enumerate() {
            let weights: Vec<_> = weights.iter().map(|ws| ws[cl]).collect();
            let mut optim = dirichlet_fit::AdamOptimizer::new(param.len());
            // let mut optim = dirichlet_fit::MomentumOptimizer::new(param.len());
            let lk: f64 = centers
                .iter()
                .zip(weights.iter())
                .map(|(c, w)| w * dirichlet_fit::dirichlet_log(c, param))
                .sum();
            *param = dirichlet_fit::fit_multiple_with(
                &[centers],
                &[(1f64, weights.clone())],
                &mut optim,
                param,
            );
            let updated: f64 = centers
                .iter()
                .zip(weights.iter())
                .map(|(c, w)| w * dirichlet_fit::dirichlet_log(c, param))
                .sum();
            trace!("OPTIM\t{:.2}->{:.2}", lk, updated);
        }
        // Update likelihood.
        let next_lk = lk(centers, &params, &fractions);
        trace!("LK\t{}\t{}", t, next_lk);
        current_lk = next_lk;
    }
    trace!("{}", current_lk);
    for ((i, w), c) in weights.iter().enumerate().zip(centers.iter()) {
        trace!("{}\t{}\t{}", i, vec2str(w), vec2str(c));
    }
}

fn logsumexp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.
    } else if xs.len() == 1 {
        xs[0]
    } else {
        let max = xs.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
        let sum = xs.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
        assert!(sum >= 0., "{:?}->{}", xs, sum);
        max + sum
    }
}

fn vec2str(xs: &[f64]) -> String {
    let xs: Vec<_> = xs.iter().map(|x| format!("{:.2}", x)).collect();
    xs.join(",")
}
