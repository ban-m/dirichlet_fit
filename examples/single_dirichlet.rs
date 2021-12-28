use rand::{Rng, SeedableRng};
use rand_distr::{Dirichlet, Distribution};
use rand_xoshiro::Xoroshiro128StarStar;
fn main() {
    let param = &[0.01, 0.02, 0.03, 0.04];
    env_logger::init();
    let seed = 923840;
    let mut rng: Xoroshiro128StarStar = SeedableRng::seed_from_u64(seed);
    for _ in 0..100 {
        let seed: u64 = rng.gen::<u64>() % 10000;
        //let seed = 3352;
        let mut rng: Xoroshiro128StarStar = SeedableRng::seed_from_u64(seed);
        let dirichlet = Dirichlet::new(param).unwrap();
        let data: Vec<Vec<f64>> = (0..200)
            .map(|_| {
                let mut xs: Vec<f64> = dirichlet.sample(&mut rng);
                xs.iter_mut().for_each(|x| *x = x.ln());
                fix(&mut xs);
                xs
            })
            .collect();
        let loss: f64 = data
            .iter()
            .map(|xs| dirichlet_fit::dirichlet_log(xs, param))
            .sum();
        let total = data.len() as f64;
        let mut sums = vec![0f64; data[0].len()];
        for xs in data.iter() {
            sums.iter_mut().zip(xs.iter()).for_each(|(s, x)| *s += x);
        }
        sums.iter_mut().for_each(|x| *x /= total);
        // fix(&mut sums);
        eprintln!("{}\t{:?}", seed, sums);
        let mut estim = estim(&sums);
        let mut buf = vec![0f64; estim.len()];
        dirichlet_fit::fit_dirichlet_only_mean(&sums, &mut estim, &mut buf);
        let pred: f64 = data
            .iter()
            .map(|xs| dirichlet_fit::dirichlet_log(xs, &estim))
            .sum();
        estim.sort_by(|x, y| x.partial_cmp(&y).unwrap());
        eprintln!("LOSS\t{}\t{}\t{}\t{:?}", seed, pred, loss, estim);
    }
}

fn fix(probs: &mut [f64]) {
    const SMALL_LOG: f64 = -30f64;
    let dim = probs.len() as f64;
    let ofs = SMALL_LOG - dim.ln();
    probs.iter_mut().for_each(|p| {
        *p = if *p < ofs {
            ofs + ((*p - ofs).exp() + 1f64).ln()
        } else {
            *p + (1f64 + (ofs - *p).exp()).ln()
        };
    });
}

fn estim(xs: &[f64]) -> Vec<f64> {
    let mut param: Vec<_> = xs
        .iter()
        .map(|&x| {
            if -2.2 < x {
                x.exp() + 0.5
            } else {
                -(x - dirichlet_fit::DIGAM_ONE).recip()
            }
        })
        .collect();
    let sum: f64 = param.iter().sum();
    param.iter_mut().for_each(|x| *x = *x * 2f64 / sum);
    param
}
